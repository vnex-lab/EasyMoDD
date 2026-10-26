import json
import re

def parse_numbered_json_file(filepath):
    """
    Parse a file with numbered JSON entries like:
    0:{...}
    1:{...}
    etc.
    """
    data = []
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Split by numbered entries (e.g., "0:", "1:", "2:")
    # Pattern matches numbers followed by colon at start of line
    entries = re.split(r'\n\d+:\{', content)
    
    for i, entry in enumerate(entries):
        if not entry.strip():
            continue
        
        # Add back the opening brace if it's not the first entry
        if i > 0:
            entry = '{' + entry
        else:
            # First entry might already have the brace
            if not entry.strip().startswith('{'):
                continue
        
        # Try to parse as JSON
        try:
            # Find the last closing brace
            last_brace = entry.rfind('}')
            if last_brace != -1:
                json_str = entry[:last_brace + 1]
                obj = json.loads(json_str)
                data.append(obj)
        except json.JSONDecodeError as e:
            print(f"Error parsing entry {i}: {e}")
            continue
    
    return data

def convert_to_chat_format(debug_data):
    """Convert code debugging data to chat format"""
    chat_data = []
    
    for item in debug_data:
        if 'original_src' in item and 'changed_src' in item:
            language = item.get('language', 'code')
            status = item.get('original_status', 'error')
            original = item.get('original_src', '')
            fixed = item.get('changed_src', '')
            
            if original and fixed:
                user_msg = f"Fix this {language} code with {status}: {original}"
                bot_msg = fixed
                chat_data.append({'user': user_msg, 'bot': bot_msg})
    
    return chat_data

if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
        debug_data = parse_numbered_json_file(filepath)
        chat_data = convert_to_chat_format(debug_data)
        
        print(f"Parsed {len(debug_data)} debug entries")
        print(f"Converted to {len(chat_data)} chat pairs")
        
        # Save as JSON
        output_file = filepath.replace('.txt', '_converted.json')
        with open(output_file, 'w') as f:
            json.dump(chat_data, f, indent=2)
        
        print(f"Saved to {output_file}")
    else:
        print("Usage: python parse_code_debug_data.py <input_file>")
