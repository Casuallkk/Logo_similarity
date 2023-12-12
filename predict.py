from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
# import xlsxwriter
import os
from siamese import Siamese
import time

if __name__ == "__main__":
    model = Siamese()
    # image_1 = input('Input image_2 filename:')
    image_1 = Image.open("testdata/06.jpg")
    pro = []
    # image = []
    # image_2 = Image.open("testdata/01.jpg")
    # probability = model.detect_image(image_1, image_2)
    im_path2 = "official"
    path_list = os.listdir(im_path2)
    for path in path_list:
        image_2 = Image.open("official/" + path)
        probability = model.detect_image(image_1, image_2)[0]
        # print(model.detect_image(image_1, image_2))
        pro.append(probability)
        # image.append(image2)
    maxx = max(pro)
    index = pro.index(maxx)
    print("similarity:", float(maxx))
    print("image:", index)
    # 返回最相似图片在official文件夹中的序号
    # img = image[index]
    # plt.subplot(1, 2, 1)
    # plt.imshow(np.array(image_1))
    # plt.subplot(1, 2, 2)
    # plt.imshow(np.array(img))
    # plt.text(-12, -12, 'Similarity:%.3f' % maxx, ha='center', va='bottom', fontsize=11)
    # plt.show()
    '''
    workbook = xlsxwriter.Workbook('result_data.xlsx')
    worksheet = workbook.add_worksheet()

    im_path2 = "testdata"
    path_list = os.listdir(im_path2)
    count = 1

    for path in path_list:
        image_2 = Image.open("testdata/"+path)
        start_time = time.time()
        probability = model.detect_image(image_1, image_2)
        end_time = time.time()
        time1 = end_time - start_time
        num = str(count)
        row = 'A' + num
        data = [probability, time1]
        worksheet.write_row(row, data)
        count = count + 1
    workbook.close()
    '''