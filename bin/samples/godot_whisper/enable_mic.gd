extends CheckBox

const DEFAULT_INPUT_DEVICE := "Default"

@onready var audio_player: AudioStreamPlayer = $"../../AudioPlayer"
@onready var mic_player: AudioStreamPlayer = $"../../MicPlayer"
@onready var audio_to_text: CaptureStreamToText = $"../../CaptureStreamToText"
@onready var mic_label: Label = $"../Label"
@onready var input_device_label: Label = get_node_or_null("../InputDeviceLabel") as Label
@onready var input_device: OptionButton = get_node_or_null("../InputDevice") as OptionButton

var _device_switch_generation := 0

func _ready():
	_update_mic_warning()
	_populate_input_devices()
	if input_device != null:
		input_device.item_selected.connect(_on_input_device_item_selected)


func start_selected_source() -> void:
	_on_toggled(button_pressed)


func _on_toggled(toggled_on: bool):
	if toggled_on:
		_restart_microphone_stream()
		mic_player.play()
		audio_player.stop()
	else:
		mic_player.stop()
		audio_player.play()


func _update_mic_warning() -> void:
	if ProjectSettings.get_setting("audio/driver/enable_input", false):
		mic_label.text = ""
	else:
		mic_label.text = "Enable Microphone in Project Settings -> audio/driver/enable_input"


func _populate_input_devices() -> void:
	if input_device == null:
		return
	input_device.clear()
	var devices := AudioServer.get_input_device_list()
	if devices.is_empty():
		input_device.add_item("No input devices")
		input_device.disabled = true
		input_device.visible = true
		if input_device_label != null:
			input_device_label.visible = true
		mic_label.text = "No microphone input devices found."
		return
	AudioServer.set_input_device(DEFAULT_INPUT_DEVICE)
	var current_device := DEFAULT_INPUT_DEVICE
	audio_to_text.set_input_device_name(current_device)
	var selected_index := 0
	input_device.add_item(DEFAULT_INPUT_DEVICE)
	for i in range(devices.size()):
		var device := devices[i]
		if device == DEFAULT_INPUT_DEVICE:
			continue
		input_device.add_item(device)
		if device == current_device:
			selected_index = input_device.item_count - 1
	input_device.disabled = false
	input_device.select(selected_index)
	var show_selector := devices.size() > 1
	input_device.visible = show_selector
	if input_device_label != null:
		input_device_label.visible = show_selector


func _on_input_device_item_selected(index: int) -> void:
	if input_device == null or input_device.disabled:
		return
	var device := input_device.get_item_text(index)
	if device.is_empty():
		return
	_device_switch_generation += 1
	var switch_generation := _device_switch_generation
	input_device.disabled = true
	mic_player.stop()
	audio_to_text.recording = false
	audio_to_text.show_status("Switching input to '" + device + "'.", false)
	AudioServer.set_input_device(device)
	audio_to_text.set_input_device_name(device)
	_update_mic_warning()
	await get_tree().create_timer(0.5).timeout
	if switch_generation != _device_switch_generation:
		return
	if input_device != null:
		input_device.disabled = false
	if button_pressed:
		_restart_microphone_stream()
		mic_player.play()
		audio_to_text.recording = true
	else:
		audio_to_text.recording = true
		audio_player.play()


func _restart_microphone_stream() -> void:
	mic_player.stream = AudioStreamMicrophone.new()
	mic_player.bus = "Record"
