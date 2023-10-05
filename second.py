import cv2
import numpy as np
from skimage.segmentation import mark_boundaries, slic
from skimage.util import img_as_float
from skimage import io
from skimage.color import rgb2lab
from sklearn.cluster import KMeans

from skimage.morphology import binary_dilation
from skimage import img_as_ubyte

def dilate(img):
    """
    Увеличивает изображение на 10 процентов с использованием операции морфологического расширения.

    Parameters:
        image (numpy.ndarray): Входное изображение.

    Returns:
        numpy.ndarray: Увеличенное изображение.
    """
    # Преобразуем изображение в двоичное (если оно не было таковым)
    binary_image = img_as_ubyte(img > 0)

    # Создаем структурный элемент для операции расширения
    selem = np.ones((11, 11))  # 11x11 структурный элемент для увеличения на 10%

    # Применяем операцию расширения
    dilated_image = binary_dilation(binary_image, selem)

    return dilated_image


def get_mean(image, mask):
    """
    Рассчитывает среднее значение для указанных пикселей в изображении.

    Parameters:
        image (numpy.ndarray): Исходное изображение.
        mask (numpy.ndarray): Бинарная маска, где 1 обозначает пиксели для учета.

    Returns:
        float: Среднее значение пикселей в изображении, учитывая маску.
    """
    # Применяем маску к изображению
    masked_image = image[mask]

    # Рассчитываем среднее значение
    mean_value = np.mean(masked_image)
    
    return mean_value

def get_BWS_pixels(img, R_bar):
    """
    Заглушка для функции, предполагается что BWS пиксели удовлетворяют условию по среднему значению красного канала.

    Parameters:
        img (numpy.ndarray): Исходное изображение.
        R_bar (float): Среднее значение красного канала для здоровой кожи.

    Returns:
        numpy.ndarray: Бинарная маска, где 1 обозначает BWS пиксели, 0 - остальные пиксели.
    """
    # Заглушка - простое условие для иллюстрации
    bws_mask = img[:, :, 0] > R_bar

    return bws_mask

def find_best_match(lab_color, color_palette):
    """
    Находит наилучшее соответствие цвета из палитры по CIELAB.

    Parameters:
        lab_color (numpy.ndarray): Цвет в пространстве CIELAB.
        color_palette (list): Палитра цветов для сравнения.

    Returns:
        numpy.ndarray: Наилучшее соответствие из палитры.
    """
    # Заглушка для поиска наилучшего соответствия в палитре
    # Вместо этого нужно использовать реальный алгоритм
    # Просто возвращаем первый цвет из палитры
    return color_palette[0]


def load_img(image_path):
    """
    Загружает изображение с указанного пути.

    Parameters:
        image_path (str): Путь к изображению.

    Returns:
        numpy.ndarray: Загруженное изображение в формате NumPy array.
    """
    # Загрузка изображения с использованием OpenCV
    img = cv2.imread(image_path)

    cv2.imwrite('6.jpg', img)
    cv2.imshow('Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Проверка на успешную загрузку
    if img is None:
        print(f"Не удалось загрузить изображение с пути: {image_path}")
        return None

    # OpenCV загружает изображение в формате BGR, преобразуем в RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img_rgb



def celebi_method(img):
  # Assume that we have the functions dilate(), get_mean() and get_BWS_pixels()
  # that implement the 10% dilation of the image, getting the mean of red channel
  # and get BWS pixels in the image.

  # Extracting border and dilating is omitted due to the manual nature
  dilated = dilate(img) #Dilating the image by 10%
  healthy_skin = ((img > [90,90,90]) & (img > img)).all(axis=2) # Mark healthy pixels
  
  # Calculate the mean of red channel values for pixels marked as healthy skin
  R_bar = get_mean(healthy_skin * img[:,:,0])  
  
  bws_pixels = get_BWS_pixels(img, R_bar)  # Get BWS pixels in img using given algorithm
  return bws_pixels

def madooei_method(img):
  # Convert from sRGB to CIELAB
  lab = rgb2lab(img)

  # Create superpixel representation
  segments = slic(img, n_segments=100, compactness=10)

  # Compute the approximate Munsell specification
  # Since we aren't given palette colors and threshold distance, we calculate a simple
  # approximation of the Munsell specification as a color palette.

  # Find the best match from color palette
  best_match = find_best_match(lab, segments)

  return best_match

# Combine methods for processing one image
def process_image_single(image_path):
  img = load_img(image_path)

  

  # Celebi method
  bws_pixels = celebi_method(img)

  # Madooei method
  best_match = madooei_method(img)

  return bws_pixels, best_match

image_path="C:\VS python projects\blue-white\1.jpg"

