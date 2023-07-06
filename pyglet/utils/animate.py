from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# from IPython.display import HTML

fig, ax = plt.subplots()

img = mpimg.imread("rendering.png")
im = plt.imshow(img)


def animate(frame_num):
    img = mpimg.imread("rotation_%04i.png" % frame_num)
    im.set_data(img)
    return im


anim = FuncAnimation(fig, animate, frames=20, interval=200)
plt.show()
