## Node that does transcribing of real time audio. It requires a bus with a [AudioEffectCapture] and a [WhisperResource] language model.
class_name CaptureStreamToText
extends SpeechToText

signal transcribed_msg(is_complete: bool, new_text: String)
signal transcribed_msg_confidence(is_complete: bool, new_text: String, confidence: float)
signal transcribed_msg_tokens(is_complete: bool, new_text: String, tokens: Array)

## Initial prompt for the transcription
## For Traditional Chinese "以下是普通話的句子。"
## For Simplified Chinese "以下是普通话的句子。"
@export var initial_prompt: String

## Flag to start/stop recording
@export var recording := true:
	set(value):
		recording = value
		if recording:
			_ready()
		else:
			thread.wait_to_finish()
	get:
		return recording

## Audio step size in milliseconds. Lower values reduce latency at the cost of stability.
@export var step_ms := 1000

## Audio window length in milliseconds. Matches whisper.cpp stream --length concept.
@export var length_ms := 5000

## Audio kept from previous step in milliseconds. Matches whisper.cpp stream --keep.
@export var keep_ms := 200

## Emit unfinished text while the current sentence is still growing.
@export var emit_partial_results := true

@export_group("Realtime Backlog")

## Maximum queued audio before dropping oldest audio. Set 0 to keep all queued audio.
@export var max_pending_audio_ms := 30000

@export_group("")

@export_group("Sentence Commit")

## Commit text when Silero VAD sees enough trailing silence after the last speech segment.
@export var commit_on_vad_silence := true

## Trailing silence required before committing text from VAD segments.
@export var commit_silence_ms := 700

## Commit text when sentence punctuation is stable across several transcriptions.
@export var commit_on_stable_punctuation := true

## Number of matching punctuated transcriptions required before punctuation commits.
@export var stable_punctuation_steps := 2

## Commit before the rolling audio window drops old words.
@export var commit_on_window_timeout := true

@export_group("")

## The record bus has to have an AudioEffectCapture at index specified by [member audio_effect_capture_index]
@export var record_bus := "Record"

## The index where the [AudioEffectCapture] is located at in the [member record_bus]
@export var audio_effect_capture_index := 0

var thread: Thread
var _old_samples := PackedFloat32Array()
var _pending_frames := PackedVector2Array()
var _audio_ms_since_commit := 0.0
var _last_committed_text := ""
var _stable_punctuation_text := ""
var _stable_punctuation_count := 0

@onready var _idx := AudioServer.get_bus_index(record_bus)
@onready var _effect_capture := (
	AudioServer.get_bus_effect(_idx, audio_effect_capture_index) as AudioEffectCapture
)


## Ready function to initialize the thread and clear buffer
func _ready() -> void:
	if Engine.is_editor_hint():
		return
	if thread and thread.is_alive():
		recording = false
		thread.wait_to_finish()
	thread = Thread.new()
	_configure_capture_buffer()
	_effect_capture.clear_buffer()
	_pending_frames.clear()
	thread.start(transcribe_thread)


## Thread function to handle transcription
func transcribe_thread() -> void:
	var last_text := ""
	while recording:
		var start_time := Time.get_ticks_msec()
		var new_frames := _collect_step_frames()
		if new_frames.size() <= 0:
			continue
		var new_samples := resample(new_frames, SpeechToText.SRC_SINC_FASTEST)
		var window := _build_stream_window(new_samples)
		_old_samples = window
		var tokens := transcribe(window, initial_prompt, 0)
		if tokens.is_empty():
			continue
		var speech_segments: Array = get_last_speech_segments()
		var full_text: String = tokens.pop_front()
		var text := _remove_special_characters(full_text)
		if text.is_empty() or _is_repetitive_hallucination(text):
			continue
		var display_tokens: Array = _display_tokens(text, tokens)
		var confidence: float = _token_confidence(display_tokens)
		var time_processing := Time.get_ticks_msec() - start_time
		_audio_ms_since_commit += float(new_samples.size()) * 1000.0 / float(SpeechToText.SPEECH_SETTING_SAMPLE_RATE)
		var commit_text := _should_commit_text(text, speech_segments, window.size())
		if commit_text:
			_old_samples = _keep_window_tail(window)
			_audio_ms_since_commit = 0.0
			_last_committed_text = text
			_reset_punctuation_stability()
		if emit_partial_results and (text != last_text or commit_text):
			call_deferred("emit_signal", "transcribed_msg", commit_text, text)
			call_deferred("emit_signal", "transcribed_msg_confidence", commit_text, text, confidence)
			call_deferred("emit_signal", "transcribed_msg_tokens", commit_text, text, display_tokens)
			print(text)
			print("Confidence " + str(confidence))
			print("Transcribe " + str(time_processing / 1000.0) + " s")
		last_text = text


func _collect_step_frames() -> PackedVector2Array:
	var mix_rate: int = ProjectSettings.get_setting("audio/driver/mix_rate")
	var step_frames := int(float(max(step_ms, 1)) * mix_rate / 1000.0)
	var max_chunk_ms := max(step_ms, length_ms - keep_ms)
	var max_chunk_frames := int(float(max(max_chunk_ms, 1)) * mix_rate / 1000.0)
	while recording:
		var available := _effect_capture.get_frames_available()
		if available > 0:
			_pending_frames.append_array(_effect_capture.get_buffer(available))
			_trim_pending_frames(mix_rate)
		if _pending_frames.size() >= step_frames:
			var chunk_frames := min(_pending_frames.size(), max_chunk_frames)
			var result := _pending_frames.slice(0, chunk_frames)
			_pending_frames = _pending_frames.slice(chunk_frames)
			return result
		OS.delay_msec(1)
	return PackedVector2Array()


func _configure_capture_buffer() -> void:
	var min_buffer_seconds := float(max(length_ms + max(step_ms, 1), 1)) / 1000.0
	if max_pending_audio_ms > 0:
		min_buffer_seconds = max(min_buffer_seconds, float(max_pending_audio_ms) / 1000.0)
	if _effect_capture.buffer_length < min_buffer_seconds:
		_effect_capture.buffer_length = min_buffer_seconds


func _trim_pending_frames(mix_rate: int) -> void:
	if max_pending_audio_ms <= 0:
		return
	var max_pending_frames := int(float(max_pending_audio_ms) * mix_rate / 1000.0)
	if _pending_frames.size() <= max_pending_frames:
		return
	push_warning("Transcription is slower than audio input. Dropping oldest queued audio.")
	_pending_frames = _pending_frames.slice(_pending_frames.size() - max_pending_frames)


func _build_stream_window(new_samples: PackedFloat32Array) -> PackedFloat32Array:
	var n_samples_new := new_samples.size()
	var effective_step_ms := max(step_ms, 1)
	var n_samples_len := int(float(max(length_ms, effective_step_ms)) * SpeechToText.SPEECH_SETTING_SAMPLE_RATE / 1000.0)
	var n_samples_keep := int(float(min(keep_ms, effective_step_ms)) * SpeechToText.SPEECH_SETTING_SAMPLE_RATE / 1000.0)
	var n_samples_take := min(_old_samples.size(), max(0, n_samples_keep + n_samples_len - n_samples_new))
	var result := PackedFloat32Array()
	if n_samples_take > 0:
		result.append_array(_old_samples.slice(_old_samples.size() - n_samples_take))
	result.append_array(new_samples)
	return result


func _keep_window_tail(window: PackedFloat32Array) -> PackedFloat32Array:
	var effective_step_ms := max(step_ms, 1)
	var n_samples_keep := int(float(min(keep_ms, effective_step_ms)) * SpeechToText.SPEECH_SETTING_SAMPLE_RATE / 1000.0)
	return window.slice(max(window.size() - n_samples_keep, 0))


func _should_commit_text(text: String, speech_segments: Array, n_samples_window: int) -> bool:
	if text == _last_committed_text:
		return false
	if commit_on_vad_silence and _has_vad_trailing_silence(speech_segments, n_samples_window):
		return true
	if commit_on_stable_punctuation and _has_stable_sentence_punctuation(text):
		return true
	if commit_on_window_timeout and _has_window_timeout():
		return true
	return false


func _has_vad_trailing_silence(speech_segments: Array, n_samples_window: int) -> bool:
	if vad_model == null or speech_segments.is_empty():
		return false
	var last_segment: Dictionary = speech_segments[speech_segments.size() - 1]
	if not last_segment.has("end"):
		return false
	var window_end_cs := float(n_samples_window) * 100.0 / float(SpeechToText.SPEECH_SETTING_SAMPLE_RATE)
	var trailing_silence_ms := (window_end_cs - float(last_segment["end"])) * 10.0
	return trailing_silence_ms >= float(max(commit_silence_ms, 0))


func _has_stable_sentence_punctuation(text: String) -> bool:
	var stripped := text.strip_edges()
	if not _ends_with_sentence_punctuation(stripped):
		_reset_punctuation_stability()
		return false
	if stripped == _stable_punctuation_text:
		_stable_punctuation_count += 1
	else:
		_stable_punctuation_text = stripped
		_stable_punctuation_count = 1
	return _stable_punctuation_count >= max(stable_punctuation_steps, 1)


func _ends_with_sentence_punctuation(text: String) -> bool:
	var index := text.length() - 1
	while index >= 0 and ")]}\"'".contains(text.substr(index, 1)):
		index -= 1
	if index < 0:
		return false
	return ".!?。！？".contains(text.substr(index, 1))


func _reset_punctuation_stability() -> void:
	_stable_punctuation_text = ""
	_stable_punctuation_count = 0


func _has_window_timeout() -> bool:
	var commit_window_ms := max(float(step_ms), float(length_ms - keep_ms))
	return _audio_ms_since_commit >= commit_window_ms


func _token_confidence(tokens: Array) -> float:
	var total := 0.0
	var count := 0
	for token in tokens:
		if token is Dictionary and token.has("confidence"):
			total += clamp(float(token["confidence"]), 0.0, 1.0)
			count += 1
	if count <= 0:
		return 0.0
	return total / count


func _display_tokens(text: String, tokens: Array) -> Array:
	var words: PackedStringArray = text.split(" ", false)
	var probs: Array = []
	for token in tokens:
		if token is Dictionary and token.has("confidence") and int(token.get("id", 0)) >= 0:
			probs.push_back(clamp(float(token["confidence"]), 0.0, 1.0))
	if words.size() <= 0:
		return []
	if probs.size() <= 0:
		probs.push_back(0.0)
	var result: Array = []
	for i in range(words.size()):
		var confidence: float = float(probs[min(i, probs.size() - 1)])
		result.push_back({
			"text": words[i],
			"confidence": confidence,
		})
	return result


## Remove special characters from the message
func _remove_special_characters(message: String) -> String:
	var special_characters := [
		{"start": "[", "end": "]"}, {"start": "<", "end": ">"}, {"start": "♪", "end": "♪"}
	]
	for special_character in special_characters:
		while message.find(special_character["start"]) != -1:
			var begin_character := message.find(special_character["start"])
			var end_character := message.find(special_character["end"], begin_character + 1)
			if end_character != -1:
				message = message.substr(0, begin_character) + message.substr(end_character + 1)
			else:
				break

	var hallucinatory_characters := [". you.", ". You."]
	for hallucination in hallucinatory_characters:
		message = message.replace(hallucination, "")
	return message.strip_edges()


func _is_repetitive_hallucination(message: String) -> bool:
	var normalized := message.to_lower()
	for character in ".!?,;:()[]{}\"'":
		normalized = normalized.replace(character, " ")
	var words := normalized.split(" ", false)
	if words.size() < 6:
		return false
	var counts := {}
	for word in words:
		counts[word] = counts.get(word, 0) + 1
	for count in counts.values():
		if count >= words.size() * 0.7:
			return true
	return false


## Handle notifications
func _notification(what: int) -> void:
	if what == NOTIFICATION_WM_CLOSE_REQUEST:
		recording = false
		if thread.is_alive():
			thread.wait_to_finish()


## Get configuration warnings for the node
func _get_configuration_warnings() -> PackedStringArray:
	if language_model == null:
		return ["You need a language model."]
	return []
