import json
import time
import re
import sys
import traceback
from flask import Flask, request, jsonify, Response
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

current_task = None
chunk_queue = []
is_generating = False

def send_sse(event_name, data_dict):
    sse_payload = f"event: {event_name}\ndata: {json.dumps(data_dict)}\n\n"
    # Don't print the raw massive string every time, just log the event
    print(f"  --> [SSE STREAM] Sent {event_name} to Claude.")
    return sse_payload

def _extract_context(text):
    """Extract OS and CWD information from the system text."""
    import re
    extracted = []
    # Look for common metadata blocks
    for tag in ['user_information', 'ADDITIONAL_METADATA']:
        match = re.search(fr'<{tag}>(.*?)</{tag}>', text, re.IGNORECASE | re.DOTALL)
        if match:
            extracted.append(match.group(0))
    
    # Also look for specific keywords often in system prompts
    if not extracted:
        # Fallback: look for "The USER's OS version is windows" etc.
        if "OS version" in text:
            extracted.append(text[:200]) # Take first 200 chars as context hint
            
    return "\n\n".join(extracted) if extracted else ""

def _strip_system_block(text):
    """Remove the leading SYSTEM INSTRUCTIONS block that Claude Code prepends to user messages."""
    import re
    # Remove everything from start up to and including "USER REQUEST:" label if present
    match = re.search(r'USER REQUEST:\s*', text, re.IGNORECASE)
    if match:
        text = text[match.end():]
    # Also strip trailing sentinel reminder Claude Code sometimes appends
    text = re.sub(r'\s*IMPORTANT:\s*End with:.*$', '', text, flags=re.IGNORECASE | re.DOTALL)
    return text.strip()


# --- ANTHROPIC COMPATIBLE ENDPOINTS (For Claude Code) ---

@app.route('/v1/messages/count_tokens', methods=['POST'])
def count_tokens():
    print("\n[CLAUDE CODE] Requested /v1/messages/count_tokens")
    return jsonify({"input_tokens": 10})

@app.route('/v1/messages', methods=['POST'])
def anthropic_messages():
    global current_task, chunk_queue, is_generating
    
    req_json = request.json or {}
    model = req_json.get('model', 'custom-browser')
    stream = req_json.get('stream', False)
    system_data = req_json.get('system', '')
    
    # Handle system prompt if it's a list of blocks
    system_text = ""
    if isinstance(system_data, list):
        for block in system_data:
            if isinstance(block, dict) and block.get('type') == 'text':
                system_text += block.get('text', '') + "\n"
            elif isinstance(block, str):
                system_text += block + "\n"
    else:
        system_text = str(system_data)

    tools = req_json.get('tools', [])
    
    print(f"\n==================================================")
    print(f"[CLAUDE CODE] Intercepted new POST /v1/messages")
    print(f"[DEBUG] Model requested: {model}")
    print(f"[DEBUG] Tools provided: {len(tools)}")
    print(f"==================================================")
    
    messages = req_json.get('messages', [])

    # Extract context from system text
    context_info = _extract_context(system_text)

    # Only forward the last user message — skip full chat history
    last_user = None
    for m in reversed(messages):
        if m.get('role') == 'user':
            last_user = m
            break

    prompt = ""
    if context_info:
        prompt += f"ENVIRONMENT CONTEXT:\n{context_info}\n\n"

    if tools:
        prompt += "AVAILABLE TOOLS:\n"
        for t in tools:
            prompt += f"- {t['name']}: {t.get('description', 'No description')}\n"
            prompt += f"  Schema: {json.dumps(t.get('input_schema', {}))}\n"
        prompt += "\n"

    if last_user:
        content = last_user.get('content', '')
        if isinstance(content, list):
            for block in content:
                if block.get('type') == 'text':
                    text = _strip_system_block(block.get('text', ''))
                    if text.strip():
                        prompt += f"USER REQUEST:\n{text.strip()}\n"
                elif block.get('type') == 'tool_result':
                    tool_content = block.get('content', '')
                    if isinstance(tool_content, list):
                        for tb in tool_content:
                            if tb.get('type') == 'text' and tb.get('text', '').strip():
                                prompt += f"\n[TOOL RESULT for {block.get('tool_use_id')}]:\n{tb['text'].strip()}\n"
                    elif isinstance(tool_content, str) and tool_content.strip():
                        prompt += f"\n[TOOL RESULT for {block.get('tool_use_id')}]:\n{tool_content.strip()}\n"
        else:
            text = _strip_system_block(content)
            if text.strip():
                prompt += f"USER REQUEST:\n{text.strip()}\n"

    # APPEND SENTINEL INSTRUCTION
    prompt += "\nREMINDER: End your FINAL response with 'response_completed' on a new line. To use a tool, use: <tool_call name=\"TOOL_NAME\" input='{\"arg\":\"val\"}' id=\"tool_123\" />"

    print(f"[FLASK] Task compiled! length: {len(prompt)} chars. Placing in queue for browser.")
    current_task = prompt
    chunk_queue = []
    is_generating = True
    SENTINEL = "response_completed"
    
    if stream:
        def generate():
            global is_generating, chunk_queue
            print("  [DEBUG] Initializing SSE Generator...")
            try:
                yield send_sse("message_start", {
                    "type": "message_start",
                    "message": {
                        "id": "msg_custom",
                        "type": "message",
                        "role": "assistant",
                        "model": model,
                        "content": [],
                        "stop_reason": None,
                        "stop_sequence": None,
                        "usage": {"input_tokens": 100, "output_tokens": 0}
                    }
                })
                
                # We start with a text block
                yield send_sse("content_block_start", {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": ""}
                })
                
                full_raw_text = ""
                sent_text_index = 0
                active_tool_ids = set()
                
                while True:
                    if chunk_queue:
                        chunk = chunk_queue.pop(0)
                        full_raw_text += chunk
                        
                        # Process for tool calls with better resilience
                        import re
                        # Find all potential <tool_call ... /> or <tool_call ... > tags
                        tag_pattern = r'<tool_call\s+(.*?)[\s/]*>'
                        matches = list(re.finditer(tag_pattern, full_raw_text, re.DOTALL | re.IGNORECASE))
                        
                        current_search_pos = sent_text_index
                        for match in matches:
                            tag_content = match.group(1)
                            tool_start = match.start()
                            
                            # Send text before the tool call
                            if tool_start > current_search_pos:
                                text_to_send = full_raw_text[current_search_pos:tool_start]
                                text_to_send = re.sub(re.escape(SENTINEL), "", text_to_send, flags=re.IGNORECASE)
                                if text_to_send:
                                    yield send_sse("content_block_delta", {
                                        "type": "content_block_delta",
                                        "index": 0,
                                        "delta": {"type": "text_delta", "text": text_to_send}
                                    })
                            
                            # Extract attributes from the tag content
                            # Using a simple attribute parser
                            attrs = {}
                            for attr_match in re.finditer(r'(\w+)=([\'"])(.*?)\2', tag_content, re.DOTALL):
                                attrs[attr_match.group(1).lower()] = attr_match.group(3)
                            
                            tool_name = attrs.get('name', '').strip()
                            # Normalizing tool name to lowercase (as expected by Claude)
                            tool_name = tool_name.lower()
                            tool_input_str = attrs.get('input', '').strip()
                            tool_id = attrs.get('id', '')
                            
                            if not tool_name or not tool_input_str:
                                # Tag incomplete or malformed, skip for now
                                continue

                            if not tool_id:
                                tool_id = f"tool_gen_{len(active_tool_ids) + 1}"
                            
                            if tool_id not in active_tool_ids:
                                active_tool_ids.add(tool_id)
                                try:
                                    clean_input = tool_input_str.strip('`').strip()
                                    tool_input = json.loads(clean_input)
                                    print(f"  [DEBUG] Tool Call Detected: {tool_name} (ID: {tool_id})")
                                    print(f"  [DEBUG] Tool Input: {json.dumps(tool_input)}")
                                    
                                    yield send_sse("content_block_start", {
                                        "type": "content_block_start",
                                        "index": len(active_tool_ids), 
                                        "content_block": {
                                            "type": "tool_use",
                                            "id": tool_id,
                                            "name": tool_name,
                                            "input": tool_input
                                        }
                                    })
                                    yield send_sse("content_block_stop", {"type": "content_block_stop", "index": len(active_tool_ids)})
                                except Exception as e:
                                    print(f"  [ERROR] Failed to parse tool input: {e}")
                                    yield send_sse("content_block_delta", {
                                        "type": "content_block_delta",
                                        "index": 0,
                                        "delta": {"type": "text_delta", "text": f"\n[BRIDGE ERROR: Malformed tool input for {tool_name}]\n"}
                                    })
                            
                            current_search_pos = match.end()
                            sent_text_index = current_search_pos

                        # If no more tools, send remaining text up to current end as delta
                        if sent_text_index < len(full_raw_text):
                            # But wait, we might be in the middle of a tool call tag
                            # Only send text that is definitely NOT part of a partial tool call tag
                            last_tag_start = full_raw_text.rfind('<tool_call')
                            if last_tag_start > sent_text_index:
                                text_upto = last_tag_start
                            else:
                                # No partial tag or the previous tags were all completed
                                text_upto = len(full_raw_text)
                            
                            # Also check for partial sentinel
                            for i in range(1, len(SENTINEL)):
                                if full_raw_text.lower().endswith(SENTINEL[:i].lower()):
                                    text_upto = min(text_upto, len(full_raw_text) - i)
                                    break

                            if text_upto > sent_text_index:
                                text_to_send = full_raw_text[sent_text_index:text_upto]
                                if text_to_send:
                                    yield send_sse("content_block_delta", {
                                        "type": "content_block_delta",
                                        "index": 0,
                                        "delta": {"type": "text_delta", "text": text_to_send}
                                    })
                                sent_text_index = text_upto

                    elif not is_generating:
                        # Final drain
                        print("  [DEBUG] Done signal received. Finalizing SSE stream.")
                        break
                    else:
                        time.sleep(0.05)

                yield send_sse("content_block_stop", {"type": "content_block_stop", "index": 0})
                yield send_sse("message_delta", {
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                    "usage": {"output_tokens": 150}
                })
                yield send_sse("message_stop", {"type": "message_stop"})
                print("  [DEBUG] SSE Stream completed.\n")
                
            except Exception as e:
                print(f"\n[CRITICAL ERROR IN SSE GENERATOR] {e}")
                traceback.print_exc()
            
        return Response(generate(), mimetype='text/event-stream', headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})
    else:
        # Provide blocking/synchronous reply
        print("  [DEBUG] Claude requested synchronous response. Blocking...")
        buffer = ""
        while True:
            if chunk_queue:
                buffer += chunk_queue.pop(0)
            elif not is_generating:
                break
            else:
                time.sleep(0.05)
                
        # Sync path: Robust two-stage parsing
        import re
        tag_pattern = r'<tool_call\s+(.*?)[\s/]*>'
        matches = list(re.finditer(tag_pattern, buffer, re.DOTALL | re.IGNORECASE))
        
        content_blocks = []
        last_pos = 0
        active_tool_ids = set()
        
        for match in matches:
            text_before = buffer[last_pos:match.start()].strip()
            if text_before:
                content_blocks.append({"type": "text", "text": text_before})
            
            tag_content = match.group(1)
            attrs = {}
            for attr_match in re.finditer(r'(\w+)=([\'"])(.*?)\2', tag_content, re.DOTALL):
                attrs[attr_match.group(1).lower()] = attr_match.group(3)
            
            tool_name = attrs.get('name', '').strip().lower()
            tool_input_str = attrs.get('input', '').strip()
            tool_id = attrs.get('id', '')
            
            if not tool_name or not tool_input_str:
                continue

            if not tool_id:
                tool_id = f"tool_gen_{len(active_tool_ids) + 1}"
            
            if tool_id not in active_tool_ids:
                active_tool_ids.add(tool_id)
                try:
                    clean_input = tool_input_str.strip('`').strip()
                    tool_input = json.loads(clean_input)
                    content_blocks.append({
                        "type": "tool_use",
                        "id": tool_id,
                        "name": tool_name,
                        "input": tool_input
                    })
                except:
                    content_blocks.append({"type": "text", "text": f"\n[BRIDGE ERROR: Malformed tool input for {tool_name}]\n"})
            
            last_pos = match.end()
        
        remaining_text = buffer[last_pos:].strip()
        if remaining_text:
            content_blocks.append({"type": "text", "text": remaining_text})

        if not content_blocks:
            content_blocks = [{"type": "text", "text": buffer}]

        print(f"  [DEBUG] Synchronous reply finished. Returning {len(content_blocks)} content blocks.")   
        return jsonify({
            "id": "msg_custom",
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": content_blocks,
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 100, "output_tokens": 150}
        })


# --- BROWSER ENDPOINTS ---
@app.route('/browser', methods=['GET'])
def browser_get():
    global current_task
    if current_task is not None:
        task = current_task
        current_task = None
        print(f"[BROWSER] Polled for task. Sent prompt to browser script.")
        return jsonify({"has_task": True, "prompt": task})
    return jsonify({"has_task": False})

@app.route('/browser/chunk', methods=['POST'])
def browser_chunk():
    global chunk_queue
    chunk = request.json.get('chunk')
    if chunk:
        print(f"[BROWSER] Chunk pushed: {repr(chunk[:50])}... (len: {len(chunk)})")
        chunk_queue.append(chunk)
    else:
        print("[BROWSER] Emtpy chunk pushed?")
    return jsonify({"status": "ok"})

@app.route('/browser/done', methods=['POST'])
def browser_done():
    global is_generating
    print(f"\n[BROWSER] Received 'DONE' signal / response_completed detected!")
    is_generating = False
    return jsonify({"status": "ok"})

if __name__ == '__main__':
    print("Real-Time Claude Code Bridge Server running on http://127.0.0.1:5000 ...")
    app.run(host='127.0.0.1', port=5000, threaded=True)