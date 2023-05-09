from glob import glob

from PIL import Image

from compas_surrogate.logger import logger


class GifGenerator:
    def __init__(self, regex=None, fname='anim.gif', duration=100, loop=0, image_fnames=None):
        self.regex = regex
        if image_fnames is None:
            image_fnames = sorted(glob(regex))
        self.image_fnames = image_fnames
        self.fname = fname
        self.duration = duration
        self.loop = loop

    @classmethod
    def make_animation(cls, regex=None, fname='anim.gif', duration=100, loop=0, image_fnames=None):
        gg = cls(regex, fname, duration=duration, loop=loop, image_fnames=image_fnames)
        gg.make_gif()

    def make_gif(self):
        images = []
        for fname in self.image_fnames:
            images.append(Image.open(fname))
        if self.loop:  # add images in reverse order
            images.extend(images[::-1])

        if len(images) > 1:

            images[0].save(
                self.fname,
                save_all=True,
                append_images=images[1:],
                duration=self.duration,
                loop=0,
                optimize=False,
            )
        else:
            logger.critical(f"No images found for {self.regex}")

        logger.info(f"Saved gif (using {len(self.image_fnames)} images) to {self.fname}.")


def make_gif(regex=None, fname='anim.gif', duration=100, loop=True, image_fnames=None):
    GifGenerator.make_animation(regex, fname, duration=duration, loop=loop, image_fnames=image_fnames)
