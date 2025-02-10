import json
from time import sleep
import cv2
from mpmath import extend
from torch.utils.data import Dataset, DataLoader
import os
import os.path
import numpy as np

class FootballDataset(Dataset):
    def __init__(self, root_list):
        # self.total_image = 0
        self.root_list = root_list
        self.video_lists = []
        self.json_files = []
        self.data_1 = []
        self.data_2 = []
        self.data_3 = []
        self.data_all = [self.data_1, self.data_2, self.data_3]
        self.json_data = {}

        self.total_image = [0, 0, 0]

        for vid_idx, root in enumerate(self.root_list):
            video_names = sorted([f for f in os.listdir(root) if f.endswith('.mp4')])
            json_files = sorted([f for f in os.listdir(root) if f.endswith('.json')])
            self.video_lists.extend([os.path.join(root, f) for f in video_names])
            self.json_files.extend([os.path.join(root, f) for f in json_files])

            for json_file_name in json_files:
                json_path = os.path.join(root, json_file_name)
                with open(json_path, 'r') as f:
                    annotation = json.load(f)
                    self.total_image[vid_idx] += len(annotation["images"])
                    for ann in annotation["annotations"]:
                        if ann["category_id"] == 4:
                            image_id = ann["image_id"]
                            bbox = ann["bbox"]
                            jersey_number = ann["attributes"].get("jersey_number", -1)
                            self.data_all[vid_idx].append((vid_idx, image_id, bbox, jersey_number))


    def __len__(self):
        total_image = self.total_image[0] + self.total_image[1] + self.total_image[2]
        return total_image

    def __getitem__(self, idx):
        len_idx_image_data_1 = self.total_image[0]
        len_idx_image_data_2 = self.total_image[0] + self.total_image[1]
        # len_image_data_3 = self.total_image[0] + self.total_image[1] + self.total_image[2]
        if idx <= len_idx_image_data_1:
            player_in_1_frame = [item for item in self.data_1 if item[1] == idx]
            video_path = self.video_lists[0]
        elif len_idx_image_data_1 < idx <= len_idx_image_data_2:
            idx = idx - len_idx_image_data_1
            player_in_1_frame = [item for item in self.data_2 if item[1] == idx]
            video_path = self.video_lists[1]
        else:
            idx = idx - len_idx_image_data_2
            player_in_1_frame = [item for item in self.data_3 if item[1] == idx]
            video_path = self.video_lists[2]

        #vid_idx, frame_idx, bbox, number = player_in_1_frame[0]
        cap = cv2.VideoCapture(video_path)
        crops = []
        for vid_idx, frame_idx, bbox, number in player_in_1_frame:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx -1)
            ret, frame = cap.read()
            if not ret:
                continue

            x, y, w, h = map(int, bbox)
            player_crop = frame[y:y + h, x:x + w]
            new_size = (480, 720)
            player_crop_resized = cv2.resize(player_crop, new_size)
            crops.append((frame, player_crop_resized, number))

        cap.release()
        return crops

    def display_crop(self, idx):
        # frame, player_crop, number = self.__getitem__(idx)
        crops = self.__getitem__(idx)
        for _, player_crop_resized, number in crops:
            if player_crop_resized is not None:
                cv2.imshow(f"Player Crop (Number: {number})", player_crop_resized)
                cv2.waitKey(0)
            cv2.destroyAllWindows()

if __name__ == "__main__":
    root_list = [
        "Match/Match_1951_1_0_subclip",
        "Match/Match_2022_3_0_subclip",
        "Match/Match_2023_3_0_subclip"
    ]
    dataset = FootballDataset(root_list)
    print(dataset.__len__())
    dataset.display_crop(1500)