import cv2
import numpy as np
import os



Path2 = './dataimage/'
files2 = os.listdir(Path2)

images = []
i = 1

for name in files2:
    if (name[-3:]!="jpg"):

        continue

    temp = cv2.imread(Path2 + name)
    if (i == 1):
        cv2.imshow('image', temp)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    print(name)
    temp = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)

    temp = cv2.resize(temp, (100, 100), interpolation=cv2.INTER_AREA)

    images.append(temp.flatten())
    if (i == 1):
        cv2.imshow('image', temp)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    i = 2

print(images)


images = np.array(images)

mu = np.mean(images)

images = images - mu
images = images.T
print(images.shape)




u, s, v = np.linalg.svd(images, full_matrices=False)

print(u.shape, s.shape, v.shape)




test = np.array(cv2.imread('./test_images/subject11.rightlight.jpg'))
cv2.imshow('image', test)
cv2.waitKey(0)
cv2.destroyAllWindows()
test = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)
test = cv2.resize(test, (100, 100), interpolation=cv2.INTER_AREA)

img = test.reshape(1, -1)
print(img)


img = img - mu

img = img.T

print(img[:][50])





test_x = np.empty(shape=(u.shape[0], u.shape[1]), dtype=np.int8)
print(test_x.shape)

for col in range(u.shape[1]):
    test_x[:, col] = img[:, 0] * u[:, col]

dot_test = np.array(test_x, dtype='int8').flatten()




dot_train = np.empty(shape=(u.shape[0] * u.shape[1], u.shape[1]), dtype=np.int8)

temp = np.empty(shape=(u.shape[0], u.shape[1]), dtype=np.int8)

for i in range(images.shape[1]):

    for c in range(u.shape[1]):
        temp[:, c] = images[:, i] * u[:, c]

    tempF = np.array(temp, dtype='int8').flatten()
    dot_train[:, i] = tempF[:]




sub = np.empty(shape=(u.shape[0] * u.shape[1], u.shape[1]))

for col in range(u.shape[1]):
    sub[:, col] = dot_train[:, col] - dot_test[:]





answer = np.empty(shape=(u.shape[1],))

for c in range(sub.shape[1]):
    answer[c] = np.linalg.norm(sub[:, c])





print(answer)



temp_ans = np.empty(shape=(u.shape[1],))
temp = np.copy(answer)

temp.sort()
check = temp[0]
print(check)


index = 0

for i in range(answer.shape[0]):
    if check == answer[i]:
        index = i

        break




folder_tr = '/dataimage/'
i = 0
print(index)
for filename in os.listdir(os.getcwd() + "/" + folder_tr):

    if index == i:
        print("The predicted face is: ", filename)
        temp = cv2.imread(Path2 + filename)
        cv2.imshow('image', temp)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        break

    else:
        i = i + 1
