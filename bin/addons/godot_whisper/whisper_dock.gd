@tool
extends EditorPlugin

const MENU_ITEM := "Whisper Models"
const WINDOW_SIZE := Vector2i(1024, 640)

var window: AcceptDialog
var content: Control


func _enter_tree() -> void:
	add_tool_menu_item(MENU_ITEM, _open_models_window)
	_create_models_window()


func _exit_tree() -> void:
	remove_tool_menu_item(MENU_ITEM)
	if window:
		window.queue_free()


func _create_models_window() -> void:
	window = AcceptDialog.new()
	window.title = "Whisper Models"
	window.ok_button_text = "Close"
	window.min_size = WINDOW_SIZE
	window.size = WINDOW_SIZE
	window.exclusive = false
	content = preload("res://addons/godot_whisper/whisper_dock.tscn").instantiate()
	window.add_child(content)
	EditorInterface.get_base_control().add_child(window)


func _open_models_window() -> void:
	if window == null:
		_create_models_window()
	window.popup_centered(WINDOW_SIZE)
