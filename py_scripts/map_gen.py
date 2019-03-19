import cv2
import numpy as np

img = cv2.imread('/home/patik/Obrázky/Map1.png', cv2.IMREAD_UNCHANGED)
kernelOpen = np.ones((5, 5))
kernelClose = np.ones((20, 20))

font = cv2.FONT_HERSHEY_SIMPLEX


def get_start(world_name):
    string = '<?xml version="1.0" ?>\n\
<sdf version="1.5">\n\
  <world name="{world_name}">\n\
    <!-- A global light source -->\n\
    <include>\n\
      <uri>model://sun</uri>\n\
    </include>\n'.format(world_name=world_name)
    return string


def get_end():
    string = '  </world>\n\
</sdf>\n'
    return string


def get_box(name, x_pose, y_pose, z_pose, r_orient, p_orient, y_orient,
            x_size, y_size, z_size):
    string = '    <model name="{name}">\n\
          <pose>{x_pose} {y_pose} {z_pose} {r_orient} {p_orient} {y_orient}</pose>\n\
          <link name="link">\n\
            <collision name="collision">\n\
              <geometry>\n\
                <box>\n\
                  <size>{x_size} {y_size} {z_size}</size>\n\
                </box>\n\
              </geometry>\n\
            </collision>\n\
            <visual name="visual">\n\
              <geometry>\n\
                <box>\n\
                  <size>{x_size} {y_size} {z_size}</size>\n\
                </box>\n\
              </geometry>\n\
              <material>\n\
                <ambient>1 0 0 1</ambient>\n\
                <diffuse>1 0 0 1</diffuse>\n\
              </material>\n\
            </visual>\n\
          </link>\n\
    </model>\n'.format(name=name, x_pose=x_pose, y_pose=y_pose, z_pose=z_pose, r_orient=r_orient, p_orient=p_orient,
                       y_orient=y_orient,
                       x_size=x_size, y_size=y_size, z_size=z_size)
    return string

def factorize_colors(step, image):
    im = np.zeros(gray.shape, dtype=np.int32)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            im[i,j] = int(image[i,j] / step) * step
    return im

world_name = "box_world.world"
path = "/home/patik/Diplomka/dp_ws/src/inputs_generator/worlds/"
world_path = path + world_name

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY_INV)
gray = cv2.inpaint(gray, mask, 5, cv2.INPAINT_NS)
gray = cv2.medianBlur(gray, 5)
gray = cv2.medianBlur(gray, 5)
gray = cv2.resize(gray, (480, 240))
gray = factorize_colors(10, gray)
horizontal_map = np.zeros(gray.shape, dtype=np.int32)
vertical_map = np.zeros(gray.shape, dtype=np.int32)
binary_map = np.zeros(gray.shape, dtype=np.int32)
for i in range(gray.shape[0]):
    for j in range(gray.shape[1]):
        index = j
        while index < gray.shape[1] and gray[i, index] == gray[i, j]:
            index += 1
        horizontal_map[i, j] = index - j
for i in range(gray.shape[0]):
    for j in range(gray.shape[1]):
        horizontal_map[i, j + 1:j + horizontal_map[i, j]] = 0
for j in range(gray.shape[1]):
    for i in range(gray.shape[0]):
        index = i
        while index < gray.shape[0] and gray[index, j] == gray[i, j] and horizontal_map[index, j] == horizontal_map[
            i, j]:
            index += 1
        vertical_map[i, j] = index - i

rectangles = []
for i in range(gray.shape[0]):
    for j in range(gray.shape[1]):
        if binary_map[i, j] == 0:
            binary_map[i: i + vertical_map[i, j], j: j + horizontal_map[i, j]] = 1
            s = vertical_map[i, j] * horizontal_map[i, j]
            rectangles.append([i, j, s, vertical_map[i, j], horizontal_map[i, j]])
print(len(rectangles))
scale = 1 / 10
with open(world_path, "w") as f:
    f.write(get_start(world_name))
    for i in range(len(rectangles)):
        if i > 600:
            break
        f.write(
            get_box("box_{}".format(str(i)), str(rectangles[i][0] * scale), str(rectangles[i][1] * scale), str(0.5),
                    str(0.0), str(0.0),
                    str(0.0), rectangles[i][3] * scale, rectangles[i][4] * scale,
                    gray[rectangles[i][0], rectangles[i][1]] * scale * scale))
    f.write(get_end())
cv2.imshow("gray", gray)
# cv2.imshow("mask", mask)
# cv2.imshow("gray", gray)
# cv2.imshow("arrray", vertical_map)
# cv2.imshow("img", img)
cv2.waitKey(0)
