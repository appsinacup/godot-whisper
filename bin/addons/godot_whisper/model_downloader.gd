@tool
extends Node

## Option button for selecting the model
@export var option_button: OptionButton

## Download button
@export var download_button: Button

## Label showing why the selected model might be useful
@export var model_info_label: Label

## VAD model names that use a different Hugging Face repository
const VAD_MODELS := ["silero-v6.2.0", "silero-v5.1.2"]

const MODEL_GUIDE_URL := "https://whisper.appsinacup.com/docs/documentation/models#multilingual-models"
const DOWNLOADED_PREFIX := "* "


func _ready() -> void:
	if option_button:
		option_button.item_selected.connect(_on_model_selected)
	_refresh_model_items()
	_update_model_info()


func _on_model_selected(_index: int) -> void:
	_update_model_info()


func _update_model_info() -> void:
	if model_info_label == null or option_button == null:
		return
	var model := _selected_model()
	model_info_label.text = _get_model_info(model)
	if download_button:
		download_button.text = "Redownload" if _has_model(model) else "Download"


func _get_model_info(model: String) -> String:
	var notes := PackedStringArray()
	if _has_model(model):
		notes.append("Already downloaded.")

	if model in VAD_MODELS:
		notes.append("Silero VAD.")
		notes.append("Use as `vad_model`.")
		notes.append("Realtime capture uses it to find speech/silence boundaries and skip silence.")
		return "\n".join(notes)

	if model.contains("large-v3-turbo"):
		notes.append("Recommended desktop default: near large-v3 quality with much better speed.")
	elif model.contains("large-v3"):
		notes.append("Best multilingual accuracy, but slow and high memory.")
	elif model.contains("medium"):
		notes.append("High quality. Use when latency is less important.")
	elif model.contains("small"):
		notes.append("Good speed/quality balance for desktop realtime.")
	elif model.contains("base"):
		notes.append("Fast. Good for mobile, web, and realtime tests.")
	elif model.contains("tiny"):
		notes.append("Fastest. Best for prototypes and low-end devices.")

	if model.contains(".en"):
		notes.append("English-only. Faster and usually more accurate for English.")
	else:
		notes.append("Multilingual. Use for non-English or mixed-language projects.")

	if model.contains("-q5") or model.contains("-q8"):
		notes.append("Quantized. Smaller download and lower memory use.")
	else:
		notes.append("Full precision. Higher memory/download size.")

	return "\n".join(notes)


func _refresh_model_items() -> void:
	for i in range(option_button.item_count):
		var model := _clean_model_name(option_button.get_item_text(i))
		option_button.set_item_text(i, _display_model_name(model))
	_update_model_info()


func _selected_model() -> String:
	if option_button == null:
		return ""
	return _clean_model_name(option_button.get_item_text(option_button.get_selected_id()))


func _clean_model_name(model: String) -> String:
	return model.trim_prefix(DOWNLOADED_PREFIX)


func _display_model_name(model: String) -> String:
	if _has_model(model):
		return DOWNLOADED_PREFIX + model
	return model


func _model_file_path(model: String) -> String:
	return "res://addons/godot_whisper/models/ggml-" + model + ".bin"


func _has_model(model: String) -> bool:
	return FileAccess.file_exists(_model_file_path(model))


## Called when the HTTP request is completed.
func _http_request_completed(
	result: int,
	response_code: int,
	_headers: PackedStringArray,
	_body: PackedByteArray,
	file_path: String
) -> void:
	# Handle unsuccessful download
	if result != HTTPRequest.RESULT_SUCCESS or response_code != 200:
		push_error("Can't download. Result: " + str(result) + " Code: " + str(response_code))
		return
	EditorInterface.get_resource_filesystem().scan()
	ResourceLoader.load(file_path, "WhisperResource", 2)
	_refresh_model_items()
	print("Download successful. Check " + file_path)


## Handle button press to start the download
func _on_button_pressed() -> void:
	var http_request := HTTPRequest.new()
	add_child(http_request)
	http_request.use_threads = true
	DirAccess.make_dir_recursive_absolute("res://addons/godot_whisper/models")
	var model: String = _selected_model()
	var file_path: String = _model_file_path(model)
	http_request.request_completed.connect(self._http_request_completed.bind(file_path))
	http_request.download_file = file_path
	var base_url: String
	if model in VAD_MODELS:
		base_url = "https://huggingface.co/ggml-org/whisper-vad/resolve/main/ggml-"
	else:
		base_url = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-"
	var url: String = base_url + model + ".bin?download=true"
	print("Downloading " + model + " from " + url)
	var error: int = http_request.request(url)
	# Handle HTTP request error
	if error != OK:
		push_error("An error occurred in the HTTP request.")


func _on_docs_button_pressed() -> void:
	OS.shell_open(MODEL_GUIDE_URL)
