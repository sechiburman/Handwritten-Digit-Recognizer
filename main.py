import pygame as pg
import tensorflow as tf
import os
import cv2 as cv

os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (0, 30)  # Set the window position
pg.init()

# Let's define some constants:
WHITE = (255, 255, 255)
GREY240 = (240, 240, 240)
GREY224 = (224, 224, 224)
GREY192 = (192, 192, 192)
GREY160 = (160, 160, 160)
GREY128 = (128, 128, 128)
GREY96 = (96, 96, 96)
GREY64 = (64, 64, 64)
GREY32 = (32, 32, 32)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
BACKGROUND_COLOR = GREY240

FPS = 120
LEFT_MARGIN = 64
TOP_MARGIN = 16
MAX_INPUTS = 5
INPUT_TOP = 2 * TOP_MARGIN + 32 + 1
INPUT_WIDTH = 280
INPUT_HEIGHT = 280
WIDTH = 2 * LEFT_MARGIN + INPUT_WIDTH * MAX_INPUTS + MAX_INPUTS - 1
HEIGHT = 784

window = pg.display.set_mode((WIDTH, HEIGHT))
pg.display.set_caption("HANDWRITTEN NUMBERS RECOGNITION")

model = tf.keras.models.load_model('cnn model.h5')


class Button:
	def __init__(
			self, x_pos, y_pos, width, height,
			idle_color, hover_color, pressed_color,
			outline_thickness, text_color,
			char_size, text='', font='arial'
	):
		self.xPos = int(x_pos)
		self.yPos = int(y_pos)
		self.width = int(width)
		self.height = int(height)
		self.idleColor = idle_color
		self.hoverColor = hover_color
		self.pressedColor = pressed_color
		self.lightOutlineColor = (255, 255, 255)
		self.darkOutlineColor = (0, 0, 0)
		self.outlineThickness = outline_thickness
		self.textColor = text_color
		self.charSize = char_size
		self.text = text
		self.font = font
		self.over = False
		self.clicked = False
		self.pressed = False

	def update(self, pos):
		self.clicked = False
		if self.xPos < pos[0] < self.xPos + self.width and self.yPos < pos[1] < self.yPos + self.height:
			self.over = True
			if pg.mouse.get_pressed()[0]:
				if not self.pressed:
					self.clicked = True
				self.pressed = True
			else:
				self.pressed = False
		else:
			self.over = False
			self.pressed = False

	def get_clicked(self):
		return self.clicked

	def get_text(self):
		return self.text

	def render(self):
		# DRAWING TOP-LEFT OUTLINE AS A RECTANGLE WHICH WILL BE ALMOST COMPLETELY COVERED:
		pg.draw.rect(window, self.lightOutlineColor, (self.xPos, self.yPos, self.width, self.height))

		# DRAWING BOTTOM-RIGHT OUTLINE AS RECTANGLE WHICH COVERS THE PREVIOUS ONE:
		t = self.outlineThickness
		pg.draw.rect(window, self.darkOutlineColor, (self.xPos + t, self.yPos + t, self.width - t, self.height - t))

		# DRAWING CENTER OF THE CHECK-BOX:
		if self.pressed:
			pg.draw.rect(
				window, self.pressedColor, (self.xPos + t, self.yPos + t, self.width - 2 * t, self.height - 2 * t)
			)
		elif self.over:
			pg.draw.rect(
				window, self.hoverColor, (self.xPos + t, self.yPos + t, self.width - 2 * t, self.height - 2 * t)
			)
		else:
			pg.draw.rect(
				window, self.idleColor, (self.xPos + t, self.yPos + t, self.width - 2 * t, self.height - 2 * t)
			)

		# DRAWING TEXT:
		if self.text != '':
			font = pg.font.SysFont(self.font, self.charSize)
			text = font.render(self.text, 1, self.textColor)
			window.blit(
				text,
				(
					(self.xPos + (self.width - text.get_width()) // 2),
					(self.yPos + (self.height - text.get_height()) // 2)
				)
			)


class InputField:
	def __init__(self, x_pos, y_pos):
		self.index = None
		self.nothing_happend_yet = True
		self.xPos = x_pos
		self.yPos = y_pos
		self.probabilities = list()
		self.digit_exists = None
		self.pre_predicted_pygame_img = None

	def set_index(self, index):
		self.index = index

	def get_inputs_for_model(self):
		file_name = 'Inputs images/' + str(self.index) + ' input.png'
		self.pre_predicted_pygame_img = pg.Surface((INPUT_WIDTH, INPUT_HEIGHT))
		self.pre_predicted_pygame_img.blit(window, (0, 0), (self.xPos, self.yPos, INPUT_WIDTH, INPUT_HEIGHT))
		pg.image.save(self.pre_predicted_pygame_img, file_name)
		img = cv.imread(file_name)
		img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
		img = cv.bitwise_not(img)
		# cv.imshow('PO BITWISE, PRZED RESIZE', img)
		# cv.waitKey(0)
		img = cv.resize(img, (28, 28))
		img = img.reshape(-1, 28, 28, 1)
		img = img.astype("float")
		img = img / 255.0
		return img

	def predict(self):
		self.nothing_happend_yet = False
		the_digit = self.get_inputs_for_model()
		self.probabilities = model.predict(the_digit)[0]

	def get_does_digit_exist(self):
		return self.digit_exists

	def get_probabilities(self):
		return self.probabilities

	def endless_render(self):
		pg.draw.rect(window, BLACK, (self.xPos - 1, self.yPos - 1, INPUT_WIDTH + 2, INPUT_HEIGHT + 2))
		pg.draw.rect(window, WHITE, (self.xPos, self.yPos, INPUT_WIDTH, INPUT_HEIGHT))

	def get_rect(self):
		if not self.nothing_happend_yet:
			file_name = 'Inputs images/' + str(self.index) + ' input.png'
			img = cv.imread(file_name)
			img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
			ret, thresh = cv.threshold(img, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
			contours, hierarchy = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
			cv.drawContours(img, contours, -1, (255, 0, 0), 3)
			for cnt in contours:
				x, y, w, h = cv.boundingRect(cnt)
				if not (x == 0 and y == 0 and w == WIDTH and h == HEIGHT):
					self.digit_exists = True
					return x, y, w, h
			self.digit_exists = False
		return None

	def render(self):
		pg.draw.rect(window, BLACK, (self.xPos - 1, self.yPos - 1, INPUT_WIDTH + 2, 1))
		pg.draw.rect(window, BLACK, (self.xPos - 1, self.yPos + INPUT_HEIGHT, INPUT_WIDTH + 2, 1))
		pg.draw.rect(window, BLACK, (self.xPos - 1, self.yPos - 1, 1, INPUT_HEIGHT + 2))
		pg.draw.rect(window, BLACK, (self.xPos + INPUT_WIDTH, self.yPos - 1, 1, INPUT_HEIGHT + 2))

		rect = self.get_rect()
		if rect is not None:
			x, y, w, h = rect
			x += self.xPos
			y += self.yPos
			# pg.draw.rect(window, MyColors.BLUE, (x, y, w, 1))
			# pg.draw.rect(window, MyColors.BLUE, (x, y + h, w, 1))
			# pg.draw.rect(window, MyColors.BLUE, (x, y, 1, h))
			# pg.draw.rect(window, MyColors.BLUE, (x + w, y, 1, h))
			# font = pg.font.SysFont('Comic Sans MS', 32)
			# text = font.render(str(np.argmax(self.probabilities)), True, MyColors.BLUE)
			# window.blit(text, (x, y - 32 - 16))

	def unpredict(self):
		self.nothing_happend_yet = True
		pg.draw.rect(window, WHITE, (self.xPos, self.yPos, INPUT_WIDTH, INPUT_HEIGHT))
		if self.pre_predicted_pygame_img is not None:
			window.blit(self.pre_predicted_pygame_img, (self.xPos, self.yPos, INPUT_WIDTH, INPUT_HEIGHT))


def init_buttons():
	buttons = list()
	x = (WIDTH - 4 * 128 - 3 * 32) / 2
	y = TOP_MARGIN
	buttons.append(
		Button(
			x + 0 * (128 + 32), y, 128, 32,
			GREY224, GREY240, GREY192, 1, BLACK, 18, "BRUSH"
		)
	)
	buttons.append(
		Button(
			x + 1 * (128 + 32), y, 128, 32,
			GREY224, GREY240, GREY192, 1, BLACK, 18, "RUBBER"
		)
	)
	buttons.append(
		Button(
			x + 2 * (128 + 32), y, 128, 32,
			GREY224, GREY240, GREY192, 1, BLACK, 18, "INCREASE"
		)
	)
	buttons.append(
		Button(
			x + 3 * (128 + 32), y, 128, 32,
			GREY224, GREY240, GREY192, 1, BLACK, 18, "DECREASE"
		)
	)
	return buttons


def init_input_fields(number_of_input_fields):
	n = number_of_input_fields
	input_fields = list()
	for i in range(int(n)):
		input_fields.append(InputField(WIDTH // 2 - (n - 1) * INPUT_WIDTH - INPUT_WIDTH // 2, INPUT_TOP))
		input_fields[-1].set_index(i)
	return input_fields


def endless_display(input_fields):
	# BACKGROUND:
	window.fill(BACKGROUND_COLOR)

	for i in input_fields:
		i.endless_render()


def set_input_fields_positions(input_fields):
	left_margin = (WIDTH - len(input_fields) * INPUT_WIDTH - len(input_fields) - 1) // 2
	index = 0
	pg.draw.rect(window, BACKGROUND_COLOR, (0, INPUT_TOP - 1, WIDTH, INPUT_HEIGHT + 2))
	for i in input_fields:
		i.xPos = left_margin + 1 + index * (INPUT_WIDTH + 1)
		i.yPos = INPUT_TOP
		i.set_index(index)
		index += 1
		i.endless_render()
	for i in input_fields:
		i.unpredict()
	for i in input_fields:
		i.predict()


def add_input(input_fields):
	if len(input_fields) < MAX_INPUTS:
		input_fields.append(InputField(-INPUT_WIDTH, -INPUT_HEIGHT))
	set_input_fields_positions(input_fields)


def remove_input(input_fields):
	if len(input_fields) > 1:
		input_fields.pop()
	set_input_fields_positions(input_fields)


def draw_round_line(start, end, color, canvas_rect, radius=12):
	if start is None or end is None:
		return
	dx = end[0] - start[0]
	dy = end[1] - start[1]

	distance = max(abs(dx), abs(dy))
	for i in range(distance):
		x = int(start[0] + float(i) / distance * dx)
		y = int(start[1] + float(i) / distance * dy)
		xx, yy, w, h = canvas_rect
		if xx - radius <= x < xx + radius + w:
			if yy - radius <= y < yy + radius + h:
				pg.draw.circle(window, color, (x, y), radius)
	else:
		x = start[0]
		y = start[1]
		xx, yy, w, h = canvas_rect
		if xx - radius <= x < xx + radius + w:
			if yy - radius <= y < yy + radius + h:
				pg.draw.circle(window, color, (x, y), radius)


def remove_files_from_inputs_images_folder():
	folder_path = 'Inputs images/'
	index = 0
	file_path = folder_path + str(index) + ' input.png'

	while os.path.isfile(file_path):
		os.remove(file_path)
		index += 1
		file_path = folder_path + str(index) + ' input.png'


def main():
	clock = pg.time.Clock()
	buttons = init_buttons()
	input_fields = init_input_fields(1)

	endless_display(input_fields)

	last_mouse_pos = None
	tool = 'BRUSH'
	draw_on = False
	run = True

	while run:
		clock.tick(FPS)
		mouse_pos = pg.mouse.get_pos()

		for event in pg.event.get():
			if event.type == pg.QUIT:
				run = False

		# UPDATING BUTTONS:
		for btn in buttons:
			btn.update(mouse_pos)

		if pg.mouse.get_pressed()[0] and not draw_on:
			for btn in buttons:
				if btn.get_clicked():
					text = btn.get_text()
					if text == 'BRUSH':
						tool = 'BRUSH'
					elif text == 'RUBBER':
						tool = 'RUBBER'
					elif text == 'INCREASE':
						add_input(input_fields)
					elif text == 'DECREASE':
						remove_input(input_fields)
					break

		if draw_on:
			if not pg.mouse.get_pressed()[0]:
				draw_on = False
				for i in input_fields:
					i.predict()
			else:
				rect = (input_fields[0].xPos, INPUT_TOP, len(input_fields) * INPUT_WIDTH, INPUT_HEIGHT)
				if tool == 'BRUSH':
					draw_round_line(mouse_pos, last_mouse_pos, BLACK, rect, 8)
				elif tool == 'RUBBER':
					draw_round_line(mouse_pos, last_mouse_pos, WHITE, rect, 32)
		else:
			if pg.mouse.get_pressed()[0]:
				x, y = mouse_pos
				if input_fields[0].xPos <= x < input_fields[0].xPos + len(input_fields) * INPUT_WIDTH:
					if INPUT_TOP <= y < INPUT_TOP + INPUT_HEIGHT:
						draw_on = True
						for i in input_fields:
							i.unpredict()

		last_mouse_pos = mouse_pos

		# REDRAWING SOME BACKGROUND TO COVER ANY EXCESS BRUSH AND ERASER RESIDUE:
		pg.draw.rect(window, BACKGROUND_COLOR, (0, 0, WIDTH, INPUT_TOP))
		pg.draw.rect(window, BACKGROUND_COLOR, (0, INPUT_TOP + INPUT_HEIGHT, WIDTH, HEIGHT - INPUT_TOP - INPUT_HEIGHT))
		pg.draw.rect(window, BACKGROUND_COLOR, (0, 0, input_fields[0].xPos - 1, HEIGHT))
		pg.draw.rect(
			window,
			BACKGROUND_COLOR,
			(
				input_fields[-1].xPos + INPUT_WIDTH + 1,
				0,
				WIDTH - (input_fields[-1].xPos + INPUT_WIDTH + 1),
				HEIGHT
			)
		)

		# DRAWING "PROBABILITY BOXES":
		left_margin = (WIDTH - len(input_fields) * INPUT_WIDTH - len(input_fields) - 1) // 2
		font = pg.font.SysFont('Comic Sans MS', 18)
		for i in range(len(input_fields)):

			# "DIGITS BOX":
			pg.draw.rect(
				window, BLACK,
				(
					left_margin + i * (INPUT_WIDTH + 1), INPUT_TOP + INPUT_HEIGHT + TOP_MARGIN - 1,
					INPUT_WIDTH + 2, 50
				)
			)
			pg.draw.rect(
				window, WHITE,
				(
					left_margin + i * (INPUT_WIDTH + 1) + 1, INPUT_TOP + INPUT_HEIGHT + TOP_MARGIN,
					INPUT_WIDTH, 48
				)
			)

			# DRAWING DIGITS-LABELS:
			for j in range(10):
				text = font.render(str(j), True, BLACK)
				window.blit(
					text,
					(
						int(left_margin + i * (INPUT_WIDTH + 1) + 1 + INPUT_WIDTH * (1 + 2 * j) // 21),
						int(INPUT_TOP + INPUT_HEIGHT + TOP_MARGIN + 48 / 2 - text.get_height() // 2),
					)
				)

			# "PREDICTIONS BOX":
			pg.draw.rect(
				window, BLACK,
				(
					left_margin + i * (INPUT_WIDTH + 1), INPUT_TOP + INPUT_HEIGHT + TOP_MARGIN + 48,
					INPUT_WIDTH + 2, INPUT_HEIGHT + 2
				)
			)
			pg.draw.rect(
				window, WHITE,
				(
					left_margin + i * (INPUT_WIDTH + 1) + 1, INPUT_TOP + INPUT_HEIGHT + TOP_MARGIN + 49,
					INPUT_WIDTH, INPUT_HEIGHT
				)
			)

			# DRAWING RECTANGLES:
			digit_exists = input_fields[i].get_does_digit_exist()
			if digit_exists:
				probabilities = input_fields[i].get_probabilities()
				if probabilities is not None:
					for j in range(10):
						height = probabilities[j]
						pg.draw.rect(
							window, BLACK,
							(
								int(left_margin + i * (INPUT_WIDTH + 1) + 1 + INPUT_WIDTH * (1 + 2 * j) // 21),
								int(INPUT_TOP + INPUT_HEIGHT + TOP_MARGIN + 50 + 15),
								int(INPUT_WIDTH // 21),
								int((INPUT_HEIGHT - 60) * height)
							)
						)
						string = str(height * 100)
						if height * 100 == 100:
							text = font.render(string[0:3], True, BLACK)
						elif height * 100 >= 10:
							text = font.render(string[0:2], True, BLACK)
						elif height * 100 >= 1:
							text = font.render(string[0:1], True, BLACK)
						elif height * 100 >= 0.5:
							text = font.render('1', True, BLACK)
						else:
							text = font.render('0', True, BLACK)
						window.blit(
							text,
							(
								int(left_margin + i * (INPUT_WIDTH + 1) + 1 + INPUT_WIDTH * (1 + 2 * j) // 21),
								int(INPUT_TOP + INPUT_HEIGHT + TOP_MARGIN + 50 + 15 + (INPUT_HEIGHT - 60) * height + 10)
							)
						)

		for btn in buttons:
			btn.render()

		for i in input_fields:
			i.render()

		pg.display.update()

	remove_files_from_inputs_images_folder()
	pg.quit()


if __name__ == '__main__':
	main()

print('Code is done, so everything works fine!')
