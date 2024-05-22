import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from numpy.random import uniform, triangular, beta
from math import pi
from pathlib import Path
from scipy.signal import convolve

# tiny error used for nummerical stability
eps = 0.1


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def norm(lst: list) -> float:
    """[summary]
    L^2 norm of a list
    [description]
    Used for internals
    Arguments:
        lst {list} -- vector
    """
    if not isinstance(lst, list):
        raise ValueError("Norm takes a list as its argument")

    if lst == []:
        return 0

    return (sum((i**2 for i in lst)))**0.5


def polar2z(r: np.ndarray, θ: np.ndarray) -> np.ndarray:
    """[summary]
    Takes a list of radii and angles (radians) and
    converts them into a corresponding list of complex
    numbers x + yi.
    [description]

    Arguments:
        r {np.ndarray} -- radius
        θ {np.ndarray} -- angle

    Returns:
        [np.ndarray] -- list of complex numbers r e^(i theta) as x + iy
    """
    return r * np.exp(1j * θ)


class Kernel(object):
    """[summary]
    Class representing a motion blur kernel of a given intensity.

    [description]
    Keyword Arguments:
            size {tuple} -- Size of the kernel in px times px
            (default: {(100, 100)})

            intensity {float} -- Float between 0 and 1.
            Intensity of the motion blur.

            :   0 means linear motion blur and 1 is a highly non linear
                and often convex motion blur path. (default: {0})

    Attribute:
    kernelMatrix -- Numpy matrix of the kernel of given intensity

    Properties:
    applyTo -- Applies kernel to image
               (pass as path, pillow image or np array)

    Raises:
        ValueError
    """

    def __init__(self, size: tuple = (100, 100), intensity: float=0):

        # checking if size is correctly given
        if not isinstance(size, tuple):
            raise ValueError("Size must be TUPLE of 2 positive integers")
        elif len(size) != 2 or type(size[0]) != type(size[1]) != int:
            raise ValueError("Size must be tuple of 2 positive INTEGERS")
        elif size[0] < 0 or size[1] < 0:
            raise ValueError("Size must be tuple of 2 POSITIVE integers")

        # check if intensity is float (int) between 0 and 1
        if type(intensity) not in [int, float, np.float32, np.float64]:
            raise ValueError("Intensity must be a number between 0 and 1")
        elif intensity < 0 or intensity > 1:
            raise ValueError("Intensity must be a number between 0 and 1")

        # saving args
        self.SIZE = size
        self.INTENSITY = intensity

        # deriving quantities

        # we super size first and then downscale at the end for better
        # anti-aliasing
        self.SIZEx2 = tuple([2 * i for i in size])
        self.x, self.y = self.SIZEx2

        # getting length of kernel diagonal
        self.DIAGONAL = (self.x**2 + self.y**2)**0.5

        # flag to see if kernel has been calculated already
        self.kernel_is_generated = False

    def _createPath(self):
        """[summary]
        creates a motion blur path with the given intensity.
        [description]
        Proceede in 5 steps
        1. Get a random number of random step sizes
        2. For each step get a random angle
        3. combine steps and angles into a sequence of increments
        4. create path out of increments
        5. translate path to fit the kernel dimensions

        NOTE: "random" means random but might depend on the given intensity
        """

        # first we find the lengths of the motion blur steps
        def getSteps():
            """[summary]
            Here we calculate the length of the steps taken by
            the motion blur
            [description]
            We want a higher intensity lead to a longer total motion
            blur path and more different steps along the way.

            Hence we sample

            MAX_PATH_LEN =[U(0,1) + U(0, intensity^2)] * diagonal * 0.75

            and each step: beta(1, 30) * (1 - self.INTENSITY + eps) * diagonal)
            """

            # getting max length of blur motion
            self.MAX_PATH_LEN = 0.75 * self.DIAGONAL * \
                (uniform() + uniform(0, self.INTENSITY**2))

            # getting step
            steps = []

            while sum(steps) < self.MAX_PATH_LEN:

                # sample next step
                step = beta(1, 30) * (1 - self.INTENSITY + eps) * self.DIAGONAL
                if step < self.MAX_PATH_LEN:
                    steps.append(step)

            # note the steps and the total number of steps
            self.NUM_STEPS = len(steps)
            self.STEPS = np.asarray(steps)

        def getAngles():
            """[summary]
            Gets an angle for each step
            [description]
            The maximal angle should be larger the more
            intense the motion is. So we sample it from a
            U(0, intensity * pi)

            We sample "jitter" from a beta(2,20) which is the probability
            that the next angle has a different sign than the previous one.
            """

            # same as with the steps

            # first we get the max angle in radians
            self.MAX_ANGLE = uniform(0, self.INTENSITY * pi)

            # now we sample "jitter" which is the probability that the
            # next angle has a different sign than the previous one
            self.JITTER = beta(2, 20)

            # initialising angles (and sign of angle)
            angles = [uniform(low=-self.MAX_ANGLE, high=self.MAX_ANGLE)]

            while len(angles) < self.NUM_STEPS:

                # sample next angle (absolute value)
                angle = triangular(0, self.INTENSITY *
                                   self.MAX_ANGLE, self.MAX_ANGLE + eps)

                # with jitter probability change sign wrt previous angle
                if uniform() < self.JITTER:
                    angle *= - np.sign(angles[-1])
                else:
                    angle *= np.sign(angles[-1])

                angles.append(angle)

            # save angles
            self.ANGLES = np.asarray(angles)

        # Get steps and angles
        getSteps()
        getAngles()

        # Turn them into a path
        ####

        # we turn angles and steps into complex numbers
        complex_increments = polar2z(self.STEPS, self.ANGLES)

        # generate path as the cumsum of these increments
        self.path_complex = np.cumsum(complex_increments)

        # find center of mass of path
        self.com_complex = sum(self.path_complex) / self.NUM_STEPS

        # Shift path s.t. center of mass lies in the middle of
        # the kernel and a apply a random rotation
        ###

        # center it on COM
        center_of_kernel = (self.x + 1j * self.y) / 2
        self.path_complex -= self.com_complex

        # randomly rotate path by an angle a in (0, pi)
        self.path_complex *= np.exp(1j * uniform(0, pi))

        # center COM on center of kernel
        self.path_complex += center_of_kernel

        # convert complex path to final list of coordinate tuples
        self.path = [(i.real, i.imag) for i in self.path_complex]

    def _createKernel(self, save_to: Path=None, show: bool=False):
        """[summary]
        Finds a kernel (psf) of given intensity.
        [description]
        use displayKernel to actually see the kernel.

        Keyword Arguments:
            save_to {Path} -- Image file to save the kernel to. {None}
            show {bool} -- shows kernel if true
        """

        # check if we haven't already generated a kernel
        if self.kernel_is_generated:
            return None

        # get the path
        self._createPath()

        # Initialise an image with super-sized dimensions
        # (pillow Image object)
        self.kernel_image = Image.new("RGB", self.SIZEx2)

        # ImageDraw instance that is linked to the kernel image that
        # we can use to draw on our kernel_image
        self.painter = ImageDraw.Draw(self.kernel_image)

        # draw the path
        self.painter.line(xy=self.path, width=int(self.DIAGONAL / 150))

        # applying gaussian blur for realism
        self.kernel_image = self.kernel_image.filter(
            ImageFilter.GaussianBlur(radius=int(self.DIAGONAL * 0.01)))

        # Resize to actual size
        self.kernel_image = self.kernel_image.resize(
            self.SIZE, resample=Image.LANCZOS)

        # convert to gray scale
        self.kernel_image = self.kernel_image.convert("L")

        # flag that we have generated a kernel
        self.kernel_is_generated = True

    def displayKernel(self, save_to: Path=None, show: bool=True):
        """[summary]
        Finds a kernel (psf) of given intensity.
        [description]
        Saves the kernel to save_to if needed or shows it
        is show true

        Keyword Arguments:
            save_to {Path} -- Image file to save the kernel to. {None}
            show {bool} -- shows kernel if true
        """

        # generate kernel if needed
        self._createKernel()

        # save if needed
        if save_to is not None:

            save_to_file = Path(save_to)

            # save Kernel image
            self.kernel_image.save(save_to_file)
        else:
            # Show kernel
            self.kernel_image.show()

    @property
    def kernelMatrix(self) -> np.ndarray:
        """[summary]
        Kernel matrix of motion blur of given intensity.
        [description]
        Once generated, it stays the same.
        Returns:
            numpy ndarray
        """

        # generate kernel if needed
        self._createKernel()
        kernel = np.asarray(self.kernel_image, dtype=np.float32)
        kernel /= np.sum(kernel)

        return kernel

    @kernelMatrix.setter
    def kernelMatrix(self, *kargs):
        raise NotImplementedError("Can't manually set kernel matrix yet")

    def applyTo(self, image, keep_image_dim: bool = False) -> Image:
        """[summary]
        Applies kernel to one of the following:

        1. Path to image file
        2. Pillow image object
        3. (H,W,3)-shaped numpy array
        [description]

        Arguments:
            image {[str, Path, Image, np.ndarray]}
            keep_image_dim {bool} -- If true, then we will
                    conserve the image dimension after blurring
                    by using "same" convolution instead of "valid"
                    convolution inside the scipy convolve function.

        Returns:
            Image -- [description]
        """
        # calculate kernel if haven't already
        self._createKernel()

        def applyToPIL(image: Image, keep_image_dim: bool = False) -> Image:
            """[summary]
            Applies the kernel to an PIL.Image instance
            [description]
            converts to RGB and applies the kernel to each
            band before recombining them.
            Arguments:
                image {Image} -- Image to convolve
                keep_image_dim {bool} -- If true, then we will
                    conserve the image dimension after blurring
                    by using "same" convolution instead of "valid"
                    convolution inside the scipy convolve function.

            Returns:
                Image -- blurred image
            """
            # convert to RGB
            image = image.convert(mode="RGB")

            conv_mode = "valid"
            if keep_image_dim:
                conv_mode = "same"

            result_bands = ()

            for band in image.split():

                # convolve each band individually with kernel
                result_band = convolve(
                    band, self.kernelMatrix, mode=conv_mode).astype("uint8")

                # collect bands
                result_bands += result_band,

            # stack bands back together
            result = np.dstack(result_bands)

            # Get image
            return Image.fromarray(result)

        # If image is Path
        if isinstance(image, str) or isinstance(image, Path):

            # open image as Image class
            image_path = Path(image)
            image = Image.open(image_path)

            return applyToPIL(image, keep_image_dim)

        elif isinstance(image, Image.Image):

            # apply kernel
            return applyToPIL(image, keep_image_dim)

        elif isinstance(image, np.ndarray):

            # ASSUMES we have an array of the form (H, W, 3)
            ###

            # initiate Image object from array
            image = Image.fromarray(image)

            return applyToPIL(image, keep_image_dim)

        else:

            raise ValueError("Cannot apply kernel to this type.")


if __name__ == '__main__':
    image = Image.open("./images/moon.png")
    image.show()
    k = Kernel()

    k.applyTo(image, keep_image_dim=True).show()
