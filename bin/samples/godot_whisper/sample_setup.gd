extends VBoxContainer

var bus_layout :AudioBusLayout = load("res://samples/godot_whisper/sample_bus_layout.tres")
@onready var audio_player: AudioStreamPlayer = $AudioPlayer
@onready var mic_player: AudioStreamPlayer = $MicPlayer
@onready var audio_to_text: CaptureStreamToText = $CaptureStreamToText
@onready var transcribe_label: RichTextLabel = $Panel/Label
@onready var input_level_bar: ProgressBar = $InputLevel/LevelBar
@onready var input_level_label: Label = $InputLevel/LevelLabel
@onready var mic_toggle: CheckBox = $HBoxContainer/MicToggle
# Called when the node enters the scene tree for the first time.
func _init():
	AudioServer.set_bus_layout(bus_layout)

func _ready():
	audio_player.bus = "Record"
	mic_player.bus = "Record"
	audio_to_text.transcribed_msg_tokens.connect(transcribe_label._on_capture_stream_to_text_transcribed_msg_tokens)
	audio_to_text.status_changed.connect(transcribe_label._on_capture_stream_to_text_status_changed)
	audio_to_text.input_level_changed.connect(_on_capture_stream_to_text_input_level_changed)
	audio_to_text.emit_current_status()
	_start_initial_source.call_deferred()


func _start_initial_source() -> void:
	await get_tree().create_timer(0.5).timeout
	if is_instance_valid(mic_toggle) and mic_toggle.has_method("start_selected_source"):
		mic_toggle.call("start_selected_source")


func _exit_tree() -> void:
	if is_instance_valid(mic_player):
		mic_player.stop()
		mic_player.stream = null
	if is_instance_valid(audio_player):
		audio_player.stop()


func _on_capture_stream_to_text_input_level_changed(peak: float, rms: float) -> void:
	var input_level: float = clamp(sqrt(max(peak, rms)) * 1.6, 0.0, 1.0)
	input_level_bar.value = input_level
	if input_level > 0.0 and input_level < 0.01:
		input_level_label.text = "<1%"
	else:
		input_level_label.text = str(int(round(input_level * 100.0))) + "%"
