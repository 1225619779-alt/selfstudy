import os
import sys

def list_files_tree(start_path='.', prefix=''):
    """
    递归打印目录树结构
    :param start_path: 起始路径
    :param prefix: 当前层级的前缀字符串（用于控制缩进和连线）
    """
    # 获取当前目录下的所有条目，并过滤掉隐藏文件（如果需要，可注释下一行以包含隐藏文件）
    entries = [e for e in os.listdir(start_path) if not e.startswith('.')]
    # 分离目录和文件，并分别排序（目录在前，文件在后，均按字母顺序）
    dirs = []
    files = []
    for entry in entries:
        full_path = os.path.join(start_path, entry)
        if os.path.isdir(full_path):
            dirs.append(entry)
        else:
            files.append(entry)
    dirs.sort(key=str.lower)
    files.sort(key=str.lower)
    sorted_entries = dirs + files

    # 遍历所有条目，根据位置决定使用的连线字符
    for i, entry in enumerate(sorted_entries):
        full_path = os.path.join(start_path, entry)
        is_last = (i == len(sorted_entries) - 1)

        # 选择合适的前缀连线
        if is_last:
            connector = '└── '
            new_prefix = prefix + '    '
        else:
            connector = '├── '
            new_prefix = prefix + '│   '

        # 打印当前条目
        print(prefix + connector + entry)

        # 如果是目录，递归处理其内容
        if os.path.isdir(full_path):
            list_files_tree(full_path, new_prefix)


if __name__ == '__main__':
    # 获取当前工作目录
    current_dir = os.getcwd()
    print(current_dir)          # 打印根目录名称（可选）
    # 也可以打印一个点表示当前目录
    # print('.')
    list_files_tree(current_dir)