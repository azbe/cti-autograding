import argparse
import copy
import multiprocessing
import os
import pdb
import random
import re
import time
import traceback

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import scipy.ndimage
import tqdm

import digit_recognizer

SIFT_FEATURES = 2000

ANSWERS_CROP_THRESH = 0.8
ANSWERS_CROP_CUT_X = (2675, 1450)
ANSWERS_START_LEFT = (695, 985)
ANSWERS_START_RIGHT = (705, 3175)
ANSWERS_INCREASE = (114.5, 140)
ANSWERS_CELL_SIZE = (60, 80)
ANSWERS_NO_START_TOP = (100, 3615)
ANSWERS_NO_START_BOT = (300, 3615)
ANSWERS_NO_CELL_SIZE = (200, 200)
ANSWERS_CHOICES = ("A", "B", "C", "D")

SUBJECT_THRESH = (10, 101, 5)
SUBJECT_THRESH_1 = (0.1, 0.91, 0.05)

BAREM_INFORMATICA_STR = "Informatica_varianta{}.txt"
BAREM_FIZICA_STR = "Fizica_varianta{}.txt"

NUM_ANSWERS = 15
NUM_VARIANTS = 2
NUM_CHOICES = 4

RESULT_CORRECT = "\033[92mCORRECT\033[0m"
RESULT_WRONG = "\033[91mWRONG\033[0m"


class Subject:
    UNKNOWN = "UNKNOWN"
    INFORMATICA = "INFORMATICA"
    FIZICA = "FIZICA"

    def __str__(self):
        return self.value


class Answers:
    def __init__(self):
        self._answers = [None for _ in range(NUM_VARIANTS * NUM_ANSWERS)]
        self.subject = Subject.UNKNOWN
        self.subject_nr = None

    def compute(self, barem):
        correct = 0
        for idx in range(NUM_VARIANTS * NUM_ANSWERS):
            correct += self[idx] == barem[idx]
        return correct
    
    def __getitem__(self, idx):
        return self._answers[idx]
    
    def __setitem__(self, idx, value):
        self._answers[idx] = value
    
    def __repr__(self):
        return "{} nr. {}\n{}".format(
            str(self.subject), self.subject_nr,
            "\n".join(["{}: {}".format(idx + 1, answer) for idx, answer in enumerate(self._answers)]))
    
    def __eq__(self, other):
        return (self.subject == other.subject) and \
               (self.subject_nr == other.subject_nr) and \
               (self._answers == other._answers)


def load_image_cv2(path, greyscale=False, noiseless=False, resize=None):
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if resize:
        image = cv2.resize(image, resize)
    if greyscale:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if noiseless:
        image = remove_noise(image, size=5)
    return image


def load_answer_file(path):
    lines = [line.strip().split() for line in open(path, "r").readlines()]
    answers = Answers()
    answers.subject = Subject.INFORMATICA if lines[0][0] == "I" else Subject.FIZICA
    answers.subject_nr = int(lines[0][1])
    for idx, ans in lines[1:-1]:
        answers[int(idx)-1] = ans
    return answers


def plot(image):
    plt.imshow(image, cmap="Greys_r")
    plt.show()


def random_string(length=32):
    return "".join([str(random.randint(0, 9)) for _ in range(length)])


def remove_noise(image, size=3):
    return scipy.ndimage.median_filter(image, size=size)


def get_border(answer, verbose):
    def border_1d(arr):
        idx = 0
        while idx < arr.size and arr[idx]: idx += 1
        while idx < arr.size and not arr[idx]: idx += 1
        start = idx
        while idx < arr.size and arr[idx]: idx += 1
        end = idx
        return (start, end)
    for thresh in np.arange(*SUBJECT_THRESH):
        mask = answer <= thresh
        if verbose: plt.imshow(answer, cmap="Greys_r"); plt.imshow(mask, alpha=0.3); plt.show()
        for thresh_1 in np.arange(*SUBJECT_THRESH_1):
            mask_h, mask_v = np.mean(mask, axis=0), np.mean(mask, axis=1)
            mask_h, mask_v = (mask_h <= thresh_1), (mask_v <= thresh_1)
            (sy, ey), (sx, ex) = border_1d(mask_v), border_1d(mask_h)
            if (ex - sx) != 0 and (ey - sy) / (ex - sx) >= 0.85 and (ey - sy) / (ex - sx) <= 1.15 and (ey - sy) * (ex - sx) >= 8000:
                return (sy, ey), (sx, ex)
    raise ValueError("Could not crop subject number.")


def normalize(image, template, sift_features=SIFT_FEATURES):
    detector = cv2.xfeatures2d.SIFT_create(sift_features)
    img_kp, img_des = detector.detectAndCompute(image, None)
    tmp_kp, tmp_des = detector.detectAndCompute(template, None)
    
    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = matcher.match(img_des, tmp_des, None)
    matches = sorted(matches, key=lambda match: match.distance)

    img_m = np.zeros((len(matches), 2), dtype=np.float32)
    tmp_m = np.zeros((len(matches), 2), dtype=np.float32)
    for idx, match in enumerate(matches):
        img_m[idx] = img_kp[match.queryIdx].pt
        tmp_m[idx] = tmp_kp[match.trainIdx].pt
    homography, _ = cv2.findHomography(img_m, tmp_m, cv2.RANSAC)
    img = cv2.warpPerspective(image, homography, template.shape[::-1])

    return img


def visualize_answers(image, answers):
    iy, ix = ANSWERS_INCREASE
    dy, dx = ANSWERS_CELL_SIZE
    image_o = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    for k, (sy, sx) in enumerate([ANSWERS_START_LEFT, ANSWERS_START_RIGHT]):
        for i in range(15):
            pts = [(int(sy + iy * i - dy/2), int(sx + ix * j - dx/2)) for j in range(NUM_CHOICES)]
            best = ANSWERS_CHOICES.index(answers[k * NUM_ANSWERS + i])
            for j, (y, x) in enumerate(pts):
                cv2.rectangle(image_o, (x, y), (x + dx, y + dy), (0, 255, 0,) if j == best else (255, 0, 0), 3)
    if answers.subject == Subject.INFORMATICA:
        sy, sx = ANSWERS_NO_START_TOP
        dy, dx = ANSWERS_NO_CELL_SIZE
        cv2.rectangle(image_o, (int(sx - dx/2), int(sy - dy/2)), (int(sx + dx/2), int(sy + dy/2)), (0, 255, 0), 3)
        plt.title("INFORMATICA nr. {}".format(answers.subject_nr))
    elif answers.subject == Subject.FIZICA:
        sy, sx = ANSWERS_NO_START_BOT
        dy, dx = ANSWERS_NO_CELL_SIZE
        cv2.rectangle(image_o, (int(sx - dx/2), int(sy - dy/2)), (int(sx + dx/2), int(sy + dy/2)), (0, 255, 0), 3)
        plt.title("FIZICA nr. {}".format(answers.subject_nr))
    else:
        plt.title("Unknown subject")
    plot(image_o)


def crop_answers(image, template, threshold):
    h, w = image.shape
    res = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    res = res[:, w//2:]
    if res.max() < threshold:
        raise ValueError
    y, x = np.where(res == res.max())
    y, x = y[0], x[0]
    x += w // 2
    dx1, dx2 = ANSWERS_CROP_CUT_X
    return image[y:, max(0, x - dx1):min(w, x + dx2)]


def scale_rotate(image, template, scale, angle):
    img = cv2.resize(image, (int(image.shape[0] * scale), int(image.shape[1] * scale)))
    h, w = img.shape
    cy, cx = h // 2, w // 2
    img = cv2.warpAffine(img, cv2.getRotationMatrix2D((cx, cy), angle, 1.0), (w, h), flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))
    res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
    y, x = np.where(res == res.max())
    y, x = y[0], x[0]
    print(scale, angle, res.max())
    return (res.max(), (y, x))


def find_scale_rotation(image, template, verbose):
    height, width = image.shape
    pool = multiprocessing.Pool(NUM_THREADS)
    results = [pool.apply_async(scale_rotate, [image, template, scale, angle]) 
               for angle in np.arange(-70, 71, 5) 
               for scale in np.arange(1.4, 2.3, 0.1)]
    while not all([result.ready() for result in results]): 
        pass
    results = [result.get() for result in results]
    pdb.set_trace()

    h, w = image.shape
    image = img[y:, :]
    return image


def get_answers(image, model, verbose):
    answers = Answers()

    iy, ix = ANSWERS_INCREASE
    dy, dx = ANSWERS_CELL_SIZE
    for k, (sy, sx) in enumerate([ANSWERS_START_LEFT, ANSWERS_START_RIGHT]):
        for i in range(NUM_ANSWERS):
            pts = [(int(sy + iy * i - dy/2), int(sx + ix * j - dx/2)) for j in range(NUM_CHOICES)]
            sums = [np.sum(image[y:y+dy, x:x+dx]) for (y, x) in pts]
            best = np.argmin(sums)
            answers[k * NUM_ANSWERS + i] = ANSWERS_CHOICES[best]

    dy, dx = ANSWERS_NO_CELL_SIZE
    crops = [image[sy-dy//2:sy+dy//2, sx-dx//2:sx+dx//2] for (sy, sx) in [ANSWERS_NO_START_TOP, ANSWERS_NO_START_BOT]]
    borders = [get_border(crop, verbose) for crop in crops]
    crops = [crop[sy:ey, sx:ex] for crop, ((sy, ey), (sx, ex)) in zip(crops, borders)]
    sums = [np.sum(crop) for crop in crops]
    best = np.argmin(sums)
    answers.subject = Subject.INFORMATICA if best == 0 else Subject.FIZICA

    best_crop = crops[best]
    best_crop = cv2.resize(best_crop, (28, 28))
    best_crop = best_crop / 255.0
    best_crop = np.reshape(best_crop, (1, 28, 28, 1))
    preds = model.predict(best_crop)
    answers.subject_nr = np.argmax(preds) + 1

    if verbose: plot(crops[best])

    return answers


def get_result(answers, barem):
    path = os.path.join(barem, (BAREM_INFORMATICA_STR if answers.subject == Subject.INFORMATICA else BAREM_FIZICA_STR).format(answers.subject_nr))
    correct_answers = load_answer_file(path)
    result = answers.compute(correct_answers)
    return result


def compare_answers(answers, groundtruth):
    gt = load_answer_file(groundtruth)
    return answers == gt


def main(images, groundtruths, barem, template, template_title, model_path, verbose):
    template = load_image_cv2(template, greyscale=True, noiseless=True)
    template_title = load_image_cv2(template_title, greyscale=True, noiseless=True)
    model = digit_recognizer.load_model(model_path)
    for idx, path in enumerate(images):
        try:
            t0 = time.time()
            image = load_image_cv2(path, greyscale=True, noiseless=True)
            try:
                cropped = crop_answers(image, template_title, ANSWERS_CROP_THRESH)
            except ValueError:
                image = normalize(image, template)
                if verbose: plot(image)
                cropped = crop_answers(image, template_title, 0.0)
            if verbose: plot(cropped)
            answers = get_answers(cropped, model, verbose)
            result = get_result(answers, barem)
            t1 = time.time()
            result_correct = ""
            if groundtruths:
                result_correct = RESULT_CORRECT if compare_answers(answers, groundtruths[idx]) else \
                                 RESULT_WRONG
            print("[{}/{}] path: \"{}\"\tscore: {} ({:.2f}/10)\ttime: {:.2f}s\t{}".format(
                idx+1, len(images), path, result, result * 3 / 10 + 1.0, t1 - t0, result_correct))
            if verbose: visualize_answers(cropped, answers)
        except Exception as ex:
            print("[{}/{}] Error for file \"{}\"\n".format(idx+1, len(images), path), traceback.format_exc())


if __name__ == "__main__":
    np.set_printoptions(suppress=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("images", type=str, nargs="+", help="Paths to input images to grade.")
    parser.add_argument("--groundtruths", type=str, nargs="+", help="Path to groundtruth files.")
    parser.add_argument("--barem", type=str, default="./data/barem", help="Path to directory of files containing the correct answers.")
    parser.add_argument("--template", type=str, default="./template.jpg", help="Path to template to use for normalizing perspective.")
    parser.add_argument("--template_title", type=str, default="./template_title.jpg", help="Path to template to use for aligning crop.")
    parser.add_argument("--model_path", type=str, default="./digit_recognizer.h5", help="Path to trained digit recognizer")
    parser.add_argument("--verbose", action="store_true")
    args, _ = parser.parse_known_args()
    main(**vars(args))