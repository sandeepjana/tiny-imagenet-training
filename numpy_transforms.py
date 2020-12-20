# https://github.com/ZohebAbai/Tiny-ImageNet-Challenge
import imgaug as ia
from imgaug import augmenters as iaa


# Defining Customized Imagedatagenerator using imgaug library
def custom_image_datagen(input_img):
  # Sometimes(0.5, ...) applies the given augmenter in 50% of all cases,
  # e.g. Sometimes(0.5, GaussianBlur(0.3)) would blur roughly every second
  # image.
  sometimes = lambda aug: iaa.Sometimes(0.5, aug)

  seq = iaa.Sequential([
      iaa.Fliplr(0.5), # horizontal flips

      # Small gaussian blur with random sigma between 0 and 0.5.
      # But we only blur about 50% of all images.
      sometimes(iaa.GaussianBlur(sigma=(0, 0.5))),

      # crop images by -10% to 20% of their height/width
      sometimes(iaa.CropAndPad(
          percent=(-0.1, 0.2),
          pad_mode=ia.ALL,
          pad_cval=(0, 255)
        )),

      # Apply affine transformations to some of the images
      # - scale to 80-120% of image height/width (each axis independently)
      # - translate by -20 to +20 relative to height/width (per axis)
      # - rotate by -45 to +45 degrees
      # - shear by -16 to +16 degrees
      # - order: use nearest neighbour or bilinear interpolation (fast)
      # - mode: use any available mode to fill newly created pixels
      #         see API or scikit-image for which modes are available
      # - cval: if the mode is constant, then use a random brightness
      #         for the newly created pixels (e.g. sometimes black,
      #         sometimes white)
      sometimes(iaa.Affine(
          scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
          translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
          rotate=(-30, 30),
          shear=(-8, 8),
          order=[0, 1],
          cval=(0, 255),
          mode=ia.ALL
      )),

      #drop 2-5% percent of the original size, leading to large dropped
      # rectangles.
      sometimes(iaa.CoarseDropout(
                        (0.03, 0.10), size_percent=(0.01, 0.03),
                        per_channel=0.2
                    )),

      # Make some images brighter and some darker.
      # In 20% of all cases, we sample the multiplier once per channel,
      # which can end up changing the color of the images.
      sometimes(iaa.Multiply((0.8, 1.2), per_channel=0.2)),

      #Improve or worsen the contrast of images.
      #Comment it out after third model run (extreme saturation)
      sometimes(iaa.ContrastNormalization((0.75, 1.5), per_channel=0.5)),
     ],
     # do all of the above augmentations in random order
     random_order = True) # apply augmenters in random order

  output_img = seq(images=input_img)
  return output_img


def get_debug_image():
    imshape = (64, 64, 3)
    g = 8
    n = 64 // g
    im = np.zeros(imshape, dtype=np.uint8)
    for y in range(n):
        for x in range(n):
            im[g*y:g*(y+1), g*x:g*(x+1)] = np.random.randint(0, 255, size=(3,))
    return im


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    im = np.stack((get_debug_image(), get_debug_image()))
    print(im.shape)    

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(8, 8))
    ax[0, 0].imshow(im[0])
    ax[0, 1].imshow(im[1])
    im_aug = custom_image_datagen(im)
    print(im_aug.shape)
    
    ax[1, 0].imshow(im_aug[0])
    ax[1, 1].imshow(im_aug[1])
    plt.show()