from reportlab.pdfgen import canvas
WIDTH = 595.25
HEIGHT = 841.89
TOP_MARGIN = 50
BOTTOM_MARGIN = 50
RIGHT_MARGIN = 30
TEXT_HEIGHT = 30
IMAGE_HEIGHT = 200

DEBUG = False

# TODO - Add functionality for regular text vs headlines
# TODO - Add functionality for text with new lines.
# TODO - Add functionality for quering images so we'll know how much to advance in y.


class PDFGenerator(object):
    def __init__(self, name):
        self.context = canvas.Canvas(name)
        self.y = HEIGHT - TOP_MARGIN

    def __enter__(self):
        if DEBUG:
            print("Object enter method has been called.")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if DEBUG:
            print("Object exit method has been called.")
        self.close()

    def write_headline(self):
        pass
        # TODO Implement

    def write_text(self, text):
        # Check if content should be in a new page.
        self.check_new_page(TEXT_HEIGHT)

        # Add content
        self.context.drawString(RIGHT_MARGIN, self.y, text)
        self.y -= TEXT_HEIGHT

    def add_image(self, image):
        # Check if content should be in a new page.
        self.check_new_page(IMAGE_HEIGHT)

        # Add content
        self.context.drawImage(image, WIDTH/4, self.y - 180, WIDTH/2, IMAGE_HEIGHT)
        self.y -= IMAGE_HEIGHT

    def check_new_page(self, content_size):
        if self.y - content_size < BOTTOM_MARGIN:
            self.start_new_page()

    def start_new_page(self):
        self.context.showPage()
        self.y = HEIGHT - TOP_MARGIN

    def close(self):
        self.context.save()
        if DEBUG:
            print("Pdf file closed")


if __name__ == "__main__":
    pdf = PDFGenerator("Sample.pdf")
    pdf.write_text("Testing pdf generaotr")
    pdf.write_text("Testing pdf generator 2nd row")
    pdf.add_image("res/chart.png")
    pdf.add_image("res/chart.png")
    pdf.add_image("res/chart.png")
    pdf.add_image("res/chart.png")
    pdf.add_image("res/chart.png")
    pdf.close()