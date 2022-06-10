import cv2
from matplotlib import pyplot as plt
"""название входной картинки менять в 3 местах, код требует доработки и оптимизации,
 приводить пдф файлы под один размер, быстроту обработки и правильная обрезка фото """


def colored_mask(img, threshold=-1):
    # Размытие для удаления мелких шумов.
    denoised = cv2.medianBlur(img, 3)
    cv2.imwrite('denoised.bmp', denoised)

    # Сохранение в ЧБ для получения маски.
    gray = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('gray.bmp', gray)

    # Получение цветной части изображения.
    adaptiveThreshold = threshold if threshold >= 0 else cv2.mean(img)[0]
    color = cv2.cvtColor(denoised, cv2.COLOR_BGR2HLS)
    mask = cv2.inRange(color, (0, int(adaptiveThreshold / 6), 60), (180, adaptiveThreshold, 255))

    # Создание маски цветной части изображения.
    dst = cv2.bitwise_and(gray, gray, mask=mask)
    cv2.imwrite('colors_mask.bmp', dst)
    return dst


# Подключение фото
img = cv2.imread('document3.jpg')
cv2.imshow('оригинал', img)

# выделяем цвет
img = colored_mask(img)

# Находим белые пиксели
img = cv2.imread('colors_mask.bmp')
img = cv2.medianBlur(img, 1)
img = cv2.medianBlur(img, 1)
cv2.imshow('Result', img)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
nonzero = cv2.findNonZero(img)
x = 0
y = 0
h = 0
w = 0

for element in nonzero:
    y += element[0][0]
    x += element[0][1]
y = y // len(nonzero)
x = x // len(nonzero)
print(y,x)
img2 = cv2.imread('document3.jpg')

cropped = img2[x-200:x, y:y+300]
# cropped = img2[x+20:x+320, y-500:y+500] размеры для document2.jpg

cv2.imwrite(f'out/{w}.jpg', cropped)



img = cv2.imread('colors_mask.bmp', 0)

img3 = cv2.imread('document3.jpg')
im3 = cv2.resize(img3, (250, 400))
img = cv2.medianBlur(img, 1)
# Initiate ORB detector
orb = cv2.ORB_create()
# find the keypoints with ORB
kp = orb.detect(img, None)
# compute the descriptors with ORB
kp, des = orb.compute(img, kp)
# draw only keypoints location,not size and orientation
img2 = cv2.drawKeypoints(img, kp, img3, color=(0, 255, 0), flags=-1)
plt.imshow(img2), plt.show()