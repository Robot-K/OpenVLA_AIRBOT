from typing import Iterator, Tuple, Any
import h5py
import glob
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from pathlib import Path
import cv2

class AirbotMix(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for example dataset."""

    VERSION = tfds.core.Version('0.0.0')
    RELEASE_NOTES = {
        '0.0.0': 'Stack_cups, test_100.',
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'image': tfds.features.Image(
                            shape=(480, 640, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Main camera RGB observation.',
                        ),
                        # 'images': tfds.features.FeaturesDict({
                        #     'image1': tfds.features.Image(
                        #         shape=(480, 640, 3),
                        #         dtype=np.uint8,
                        #         encoding_format='png',
                        #         doc='Main camera1 RGB observation.',
                        #     ),
                        #     'image2': tfds.features.Image(
                        #         shape=(480, 640, 3),
                        #         dtype=np.uint8,
                        #         encoding_format='png',
                        #         doc='Main camera2 RGB observation.',
                        #     ),
                        # }),
                        'state': tfds.features.Tensor(
                            shape=(7,),
                            dtype=np.float32,
                            doc='Robot state, consists of [6x robot joint angles, '
                                '1x gripper position].',
                        )
                    }),
                    'action': tfds.features.Tensor(
                        shape=(7,),
                        dtype=np.float32,
                        doc='Action taken by the robot.',
                    ),
                    'language_instruction': tfds.features.Text(
                        doc='Language Instruction.'
                    ),
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'file_path': tfds.features.Text(
                        doc='Path to the original data file.'
                    ),
                }),
            }))

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Define data splits."""
        data_splits = ['data/Stack_cups']
        train_ratio = 0.8
        train_paths = []
        val_paths = []
        for split in data_splits:
            episode_paths = sorted(Path(split).glob('*.hdf5'))
            train_split = int(len(episode_paths) * train_ratio)
            train_paths.extend(episode_paths[:train_split])
            val_paths.extend(episode_paths[train_split:])
        
        return {
            'train': self._generate_examples(train_paths),
            'val': self._generate_examples(val_paths),
        }

    def _generate_examples(self, episode_paths) -> Iterator[Tuple[str, Any]]:
        """Generator of examples for each split."""

        def _parse_example(episode_path: Path) -> Tuple[str, Any]:
            # Load raw data and assemble episode
            episode = []
            
            with h5py.File(episode_path, "r") as root:
                eef_pose = root['/actions/eef_pose'][()]
                action_last_column = root['/action'][()][:, -1:]
                actions = np.hstack((eef_pose, action_last_column))
                states = root['/observations/qpos'][()]
                cams = list(root['observations/images'].keys())
                images = root[f'observations/images/{cams[0]}'][()]
                # images1 = root[f'observations/images/{cams[0]}'][()]
                # images2 = root[f'observations/images/{cams[1]}'][()]

                instruction_path = episode_path.with_name('instruction.txt')
                with open(instruction_path, 'r') as note_file:
                    language_instruction = note_file.read().strip()
                    
                for i in range(len(actions)):
                    image = cv2.imdecode(images[i], cv2.IMREAD_UNCHANGED)
                    episode.append({
                        'observation': {
                            'image': image,
                            'state': states[i],
                        },
                        'action': actions[i],
                        'language_instruction': language_instruction,
                    })
                
                # for i in range(len(actions)):
                #     image1 = cv2.imdecode(images1[i], cv2.IMREAD_UNCHANGED)
                #     image2 = cv2.imdecode(images2[i], cv2.IMREAD_UNCHANGED)
                #     episode.append({
                #         'observation': {
                #             'images': {
                #                 'image1': image1,
                #                 'image2': image2,
                #             },
                #             'state': states[i],
                #         },
                #         'action': actions[i],
                #         'language_instruction': language_instruction,
                #     })
                    
                # Create output data sample
                sample = {
                    'steps': episode,
                    'episode_metadata': {
                        'file_path': str(episode_path),
                    }
                }

            return str(episode_path), sample

        for episode_path in episode_paths:
            print(episode_path)
            yield _parse_example(episode_path)
