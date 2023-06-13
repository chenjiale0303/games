import re
import os
import sys


# 获取文件的绝对路径并将其转化为linux形式的路径
def parse_path(path: str):
    # os.path.exists(path)
    if not os.path.isfile(path):
        print("file {0} is not exist".format(path))
        exit(-1)
    path = os.path.abspath(path)
    regex = re.compile(r"\\+")
    path = re.split(regex, path)
    path = '/'.join(path)
    index = path.rfind("/")
    file_name = path[index + 1:]
    dir_name = path[0:index + 1]
    return dir_name, file_name


def delete_useless_markdown_pic(path: str):
    dir_name, file_name = parse_path(path)
    with open(path, "r", encoding="utf-8") as f:
        s = f.readlines()
        s = "".join(s)
        # .*? 匹配符号内的所有字符
        # https://blog.csdn.net/m0_37696990/article/details/105925940 markdown内容提取
        results_img = re.findall(r'!\[.*?\]\((.*?)\)', s)  # 提最述与rul
        results_src = re.findall(r'<img\s*(.*?)\s*/>', s)
        print("url img")
        print(results_src)
        set_pic = set()
        set_asset = set()
        asset = file_name.split(".")[0] + ".assets"
        asset_path = dir_name + asset
        for result in results_img:
            if result.split("/")[0] == asset:
                set_pic.add(result.split("/")[-1])
        for result in results_src:
            result = result.split("alt")[0].strip().split("=")[-1].strip().strip('"')
            if result.split("/")[0] == asset:
                set_pic.add(result.split("/")[-1])
        if len(set_pic) == 0:
            return

        print(set_pic)
        dirs = os.listdir(asset_path)
        for i in dirs:
            set_asset.add(i)

        set_delete = set_asset - set_pic
        for i in set_delete:
            os.remove(asset_path + "/" + i)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("usage like this:python markdown_pic.py xx.md")
        exit(-1)
    markdown = sys.argv[1]
    dirname, filename = parse_path(markdown)
    print(dirname)
    print(filename)
    delete_useless_markdown_pic(markdown)

