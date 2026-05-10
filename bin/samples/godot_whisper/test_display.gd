extends RichTextLabel


func update_text() -> void:
	text = completed_text + partial_text

func _process(_delta: float) -> void:
	update_text()

var completed_text := ""
var partial_text := ""

## is_complete: true when a sentence/window boundary is committed, false when still in progress.
func _on_capture_stream_to_text_transcribed_msg(is_complete: bool, new_text: String) -> void:
	_on_capture_stream_to_text_transcribed_msg_confidence(is_complete, new_text, 1.0)


func _on_capture_stream_to_text_transcribed_msg_confidence(is_complete: bool, new_text: String, confidence: float) -> void:
	var colored_text: String = _confidence_color(new_text, confidence)
	if is_complete:
		completed_text += colored_text + " "
		partial_text = ""
	else:
		if new_text != "":
			partial_text = colored_text


func _on_capture_stream_to_text_transcribed_msg_tokens(is_complete: bool, new_text: String, tokens: Array) -> void:
	var colored_text: String = _tokens_color(new_text, tokens)
	if is_complete:
		completed_text += colored_text + " "
		partial_text = ""
	else:
		if new_text != "":
			partial_text = colored_text


func _tokens_color(message: String, tokens: Array) -> String:
	if tokens.is_empty():
		return _confidence_color(message, 1.0)
	var parts := PackedStringArray()
	for token in tokens:
		if token is Dictionary and token.has("text"):
			parts.push_back(_confidence_color(str(token["text"]), float(token.get("confidence", 0.0))))
	if parts.is_empty():
		return _confidence_color(message, 1.0)
	return " ".join(parts)


func _confidence_color(message: String, confidence: float) -> String:
	var t: float = clamp(confidence, 0.0, 1.0)
	var red := 255
	var green := int(255.0 * t)
	var blue := int(255.0 * t)
	var color: String = "#%02x%02x%02x" % [red, green, blue]
	return "[color=" + color + "]" + message.xml_escape() + "[/color]"
