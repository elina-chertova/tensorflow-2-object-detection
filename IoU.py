import pandas as pd
import xml.etree.ElementTree as ET
import glob
from ast import literal_eval


class IoU:
    def __init__(self, path='result/outputs.csv'):
        self.path = path

    def xml_get_bnd_boxes(self, path):
        """Создает из xml файлов датафрейм из координат
        :param path: Путь к xml файлу
        :return: Датафрейм из координат
        """
        xml_list = []
        for xml_file in glob.glob(path):
            tree = ET.parse(xml_file)
            root = tree.getroot()
            for member in root.findall('object'):
                try:
                    value = (
                        int(member[4][0].text),
                        int(member[4][1].text),
                        int(member[4][2].text),
                        int(member[4][3].text)
                    )
                    xml_list.append(value)
                except:
                    pass
        column_name = ['xmin', 'ymin', 'xmax', 'ymax']
        xml_df = pd.DataFrame(xml_list, columns=column_name)
        return xml_df


    def bb_intersection_over_union(self, detected, real):
        """Считает метрику IoU между двумя детекциями.

        :param detected: Датафрейм из предсказанных координат определенного объекта
        :param real: Датафрейм из реальных координат определенного объекта
        :return: метрика IoU
        """
        xA = max(detected[0], real[0])
        yA = max(detected[1], real[1])
        xB = min(detected[2], real[2])
        yB = min(detected[3], real[3])
        inter_area = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        box_detected = (detected[2] - detected[0] + 1) * (detected[3] - detected[1] + 1)
        box_real = (real[2] - real[0] + 1) * (real[3] - real[1] + 1)
        iou = inter_area / float(box_detected + box_real - inter_area)
        return iou

    def __call__(self):
        df_detected = pd.read_csv(self.path, converters={'output': literal_eval})
        c = 0
        iou_score = 0
        for row in df_detected['image']:
            detected_box = pd.DataFrame({'1': [], '2': [], '3': [], '4': []})

            coord = df_detected.loc[df_detected['image'] == row]['output']
            for r in coord.item():
                detected_box.loc[len(detected_box)] = [int(r[1]), int(r[2]), int(r[3]), int(r[4])]
            # print(df_coord)
            for i in range(detected_box.shape[0]):
                c += 1

                xml_df = self.xml_get_bnd_boxes(row.split('.')[0] + '.xml')
                all_diff_boxes = []
                for j in range(xml_df.shape[0]):
                    all_diff_boxes.append(self.bb_intersection_over_union(detected_box.loc[i], xml_df.loc[j]))
                iou_score += max(all_diff_boxes)
        return iou_score / c

# i = IoU()
# k = i()
# print(k)
# print(f'IoU score for SSD = {iou_score / c}')

# print(df_detected['image'])

        # try:
        #
        #     print(row.split('.')[0] + '.xml')
        #     iou_score += bb_intersection_over_union(detected_box.loc[i], xml_df.loc[i])
        #     print('xml_df ', xml_df)
        #     print('detected_box.loc[i] ', detected_box.loc[i])
        #     print(bb_intersection_over_union(detected_box.loc[i], xml_df.loc[i]))
        # except KeyError:
        #     print('Extra box')