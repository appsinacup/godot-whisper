@tool
extends Node

## Option button for selecting the model
@export var option_button: OptionButton

## VAD model names that use a different Hugging Face repository
const VAD_MODELS := ["silero-v6.2.0", "silero-v5.1.2"]


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
	print("Download successful. Check " + file_path)


## Handle button press to start the download
func _on_button_pressed() -> void:
	var http_request := HTTPRequest.new()
	add_child(http_request)
	http_request.use_threads = true
	DirAccess.make_dir_recursive_absolute("res://addons/godot_whisper/models")
	var model: String = option_button.get_item_text(option_button.get_selected_id())
	var file_path: String = "res://addons/godot_whisper/models/ggml-" + model + ".bin"
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
