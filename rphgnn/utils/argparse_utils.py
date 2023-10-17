def parse_bool(bool_str):
    if bool_str == "True":
        return True
    elif bool_str == "False":
        return False
    else:
        raise Exception("wrong bool_str: ", bool_str)