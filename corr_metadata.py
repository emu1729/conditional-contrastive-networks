import os
import numpy as np
import pandas as pd
import random

BASE_SHAPES = ["triangle", "square", "plus", "circle", "tee", "rhombus", "pentagon", "star", "fivesquare", "trapezoid"]

BASE_TEXTURES = ["solid", "stripes", "grid", "hexgrid", "dots", "noise", "triangles", "zigzags", "rain", "pluses"]

train_shapes = ["triangle", "square", "plus", "circle", "tee"]

train_textures = ["solid", "stripes", "grid", "hexgrid", "dots"]

val_shapes = ["rhombus", "pentagon", "star", "fivesquare", "trapezoid"]

val_textures = ["noise", "triangles", "zigzags", "rain", "pluses"]

correlations = {"triangle": "solid", "square": "stripes", "plus": "grid", "circle": "hexgrid", "tee": "dots",
                "rhombus": "noise", "pentagon": "triangles", "star": "zigzags", "fivesquare": "rain",
                "trapezoid": "pluses"}


def create_metadata(data_folder, corr):
    print(corr)
    metadata = {'filename': []}
    all_images = os.listdir(data_folder)

    num_corr = round(700 * corr)
    num_uncorr = round(700 * (1 - corr))

    for shape in train_shapes:
        corr_name = shape + '_' + correlations[shape]
        corr_images = [i for i in all_images if (i.split('_')[0] + "_" + i.split('_')[1]) == corr_name]
        uncorr_images = [i for i in all_images if ((i.split('_')[0] == shape and i.split('_')[1] in train_textures)
                                                   and (i.split('_')[0] + "_" + i.split('_')[1] != corr_name))]

        assert len(corr_images) == 1000
        assert len(uncorr_images) == 4000

        metadata['filename'] += random.sample(corr_images, k=num_corr)
        metadata['filename'] += random.sample(uncorr_images, k=num_uncorr)

    df = pd.DataFrame.from_dict(metadata)
    df['shape_t'] = df.apply(lambda x: x['filename'].split('_')[0], axis=1)
    df['texture_t'] = df.apply(lambda x: x['filename'].split('_')[1], axis=1)
    df['split'] = 'train'

    df.to_csv('/data/ddmg/xray_data/trifeature/metadata_corr' + str(corr) + '.csv')


def create_val_metadata(data_folder):
    metadata = {'filename': []}
    all_images = os.listdir(data_folder)

    metadata['split'] = ['train', 'val', 'test'] * 3000

    images = [i for i in all_images if (i.split('_')[0] in val_shapes and i.split('_')[1] in val_textures)]

    metadata['filename'] = random.sample(images, k=9000)

    df = pd.DataFrame.from_dict(metadata)
    df['shape_v'] = df.apply(lambda x: x['filename'].split('_')[0], axis=1)
    df['texture_v'] = df.apply(lambda x: x['filename'].split('_')[1], axis=1)

    df.to_csv('/data/ddmg/xray_data/trifeature/metadata_val.csv')


if __name__ == '__main__':
    for i in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]:
       create_metadata('/data/ddmg/xray_data/trifeature/color_texture_shape_stimuli', i)
    #create_val_metadata('/data/ddmg/xray_data/trifeature/color_texture_shape_stimuli')










