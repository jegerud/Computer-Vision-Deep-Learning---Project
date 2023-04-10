import numpy as np

labels_code = ["D00", "D10", "D20", "D40"]

def area_between_points(box):
    xmin = box[0]
    ymin = box[1]
    xmax = box[2]
    ymax = box[3]

    return (xmax-xmin)*(ymax-ymin)


def get_center(box):
    xmin = box[0]
    ymin = box[1]
    xmax = box[2]
    ymax = box[3]

    return (xmax-xmin, ymax-ymin)

def get_mean_box_area(dataloader, subsamples):
    mean_box = {
        "D00": [],
        "D10": [],
        "D20": [],
        "D40": []
    }

    for subsample in subsamples:
        boxes = dataloader.dataset._get_annotation(subsample)[0]
        labels = dataloader.dataset._get_annotation(subsample)[1]
        for i in range(len(labels)):
            mean_box[labels_code[labels[i]-1]].append(area_between_points(boxes[i]))
            # print(f"{labels_code[labels[i]-1]}: {area_between_points(boxes[i])}")
    for key in mean_box:
        mean_box[key] = sum(mean_box[key]) / len(mean_box[key])

    return mean_box

def get_number_of_boxes(dataloader, subsamples):
    count_box = {
        "D00": 0,
        "D10": 0,
        "D20": 0,
        "D40": 0
    }

    for subsample in subsamples:
        labels = dataloader.dataset._get_annotation(subsample)[1]
        for i in range(len(labels)):
            count_box[labels_code[labels[i]-1]] += 1

    return count_box


def get_label_mean_center(dataloader, subsamples):
    mean_center = {
        "D00": [[],[]],
        "D10": [[],[]],
        "D20": [[],[]],
        "D40": [[],[]]
    }

    for subsample in subsamples:
        boxes = dataloader.dataset._get_annotation(subsample)[0]
        labels = dataloader.dataset._get_annotation(subsample)[1]
        for i in range(len(labels)):
            x, y = get_center(boxes[i])
            mean_center[labels_code[labels[i]-1]][0].append(x)
            mean_center[labels_code[labels[i]-1]][1].append(y)
            # print(f"{labels_code[labels[i]-1]}: {area_between_points(boxes[i])}")
    for key in mean_center:
        x_mean = sum(mean_center[key][0]) / len(mean_center[key][0])
        y_mean = sum(mean_center[key][1]) / len(mean_center[key][1])
        mean_center[key] = (x_mean, y_mean)

    # print(mean_center)

    return mean_center