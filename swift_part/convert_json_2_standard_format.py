import os
import json


def is_standard_format(item):
    """判断单条数据是否为标准格式"""
    # 检查核心字段是否为标准名称
    if "image" in item and "images" not in item:
        return False  # 存在旧字段image但无新字段images，非标准
    if "conversations" in item and "messages" not in item:
        return False  # 存在旧字段conversations但无新字段messages，非标准

    # 检查messages结构是否标准
    messages = item.get("messages", [])
    for msg in messages:
        if "from" in msg and "role" not in msg:
            return False  # 存在旧字段from但无新字段role，非标准
        if "value" in msg and "content" not in msg:
            return False  # 存在旧字段value但无新字段content，非标准

        # 检查角色是否为标准值
        role = msg.get("role", "")
        if role not in ["user", "assistant"]:
            return False  # 角色不是user/assistant，非标准

    return True


def transform_single_item(original_item):
    """转换单条数据的格式（仅对非标准格式生效）"""
    # 如果已是标准格式，直接返回
    if is_standard_format(original_item):
        # 处理标准格式中可能存在的空images字段（如果是纯文本）
        if "images" in original_item and original_item["images"] is None:
            original_item.pop("images", None)
        return original_item

    # 初始化转换后的数据（不包含images字段）
    transformed = {
        "id": original_item.get("id", None),
        "messages": []
    }

    # 只有当原始数据有image字段时，才添加images字段（处理图文数据）
    if "image" in original_item:
        transformed["images"] = original_item["image"]

    # 处理conversations到messages的转换（过滤system角色）
    for conv in original_item.get("conversations", []):
        role = conv.get("from", "")
        if role == "system":
            continue  # 跳过system角色
        if role == "human":
            role = "user"
        elif role == "gpt":
            role = "assistant"

        transformed_msg = {
            "role": role,
            "content": conv.get("value", "")  # 替换value为content
        }
        transformed["messages"].append(transformed_msg)

    # 移除id为None的字段（可选，根据实际需求决定是否保留）
    if transformed["id"] is None:
        transformed.pop("id", None)

    return transformed


def process_json_file(file_path):
    """处理单个JSON文件（仅处理非标准格式内容）"""
    try:
        # 读取原始JSON数据
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 确保数据是列表格式
        if not isinstance(data, list):
            data = [data]

        # 检查是否所有数据都已是标准格式
        all_standard = all(is_standard_format(item) for item in data)
        if all_standard:
            # 即使是标准格式，也检查并移除纯文本数据的空images字段
            need_update = False
            for item in data:
                if "images" in item and item["images"] is None:
                    item.pop("images", None)
                    need_update = True
            if need_update:
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=4)
                print(f"文件 {file_path} 已是标准格式，已移除纯文本数据的空images字段")
            else:
                print(f"文件 {file_path} 已是标准格式，跳过处理")
            return

        # 仅转换非标准格式的数据
        transformed_data = [transform_single_item(item) for item in data]

        # 写入转换后的数据
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(transformed_data, f, ensure_ascii=False, indent=4)

        print(f"成功处理: {file_path}")
    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {str(e)}")


def traverse_and_process(folder_path):
    """遍历文件夹及子文件夹处理所有JSON文件"""
    if not os.path.isdir(folder_path):
        print(f"错误: {folder_path} 不是有效的文件夹路径")
        return

    # 遍历所有文件和子文件夹
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith('.json'):
                file_path = os.path.join(root, file)
                process_json_file(file_path)


if __name__ == "__main__":
    target_folder = '/data/scy/SCY/SonoVLM_V2/dataset/echovlm'
    traverse_and_process(target_folder)
    print("所有JSON文件处理完成!")