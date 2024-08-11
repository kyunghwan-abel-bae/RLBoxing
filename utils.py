bprint_temp = [""]
def bprint(msg):
    print(msg)
    bprint_temp[0] = bprint_temp[0] + msg + "\n"

def save_bprint(str_filename):
    with open(str_filename, "a") as file:
        file.write(bprint_temp[0])
    bprint_temp[0] = ""

def clear_bprint():
    bprint_temp[0] = ""