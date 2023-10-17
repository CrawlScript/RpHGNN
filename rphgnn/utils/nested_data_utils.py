# coding=utf-8


def nested_map(data, func):
    if isinstance(data, list):
        return [nested_map(item, func) for item in data]
    else:
        return func(data)

def gather_h_y(target_h_list_list, y, index):
    def func(data):
        return data[index]

    target_h_list_list_, y_ = nested_map(target_h_list_list, func), nested_map(y, func)

    return target_h_list_list_, y_


def nested_gather(nested_data, index):
    def func(data):
        return data[index]
    return nested_map(nested_data, func)
