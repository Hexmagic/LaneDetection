import os


def convertImg(line):
    line = line.replace("/root/data/LaneSeg/Image_Data/", "")
    road, suff = line.split('/', 1)
    lin = f'D:/Compressed/Image_Data/{road}/ColorImage_{road.lower()}/ColorImage/{suff}'
    return lin


def convertLabel(line):
    line = line.replace("/root/data/LaneSeg/Gray_Label",
                        "D:/Compressed/Gray_Label")
    return line


def main():
    for filename in os.listdir('data_list'):
        first = True
        with open(os.path.join('data_list', filename), 'r') as f:
            with open(os.path.join('data_list_2', filename), 'w') as f2:
                for line in f:
                    if first:
                        first = False
                        f2.write('img,label\n')
                        continue
                    img, label = line.split(',')
                    img = convertImg(img)
                    label = convertLabel(label)
                    f2.write(','.join([img, label]))


if __name__ == "__main__":
    main()