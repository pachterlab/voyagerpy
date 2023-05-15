import argparse
from bs4 import BeautifulSoup, Doctype
import pathlib


def parse_html_file(input_file, pretty=False):
    # Parse the input HTML file
    with open(input_file) as file:
        html_doc = file.read()
    soup = BeautifulSoup(html_doc, "html.parser")

    # Remove the <head> and <html> elements
    head_tag = soup.head.extract()
    html_tag = soup.html.unwrap()

    # Rename the <body> tag to <div>
    body_tag = soup.body
    body_tag.name = "div"

    # TODO: remove class anchor-link

    # Get all the img tags
    img_tags = soup.find_all("img")
    for img_tag in img_tags:
        # append /voyagerpy/img to the src attribute
        img_tag["src"] = "/voyagerpy/img/" + img_tag["src"]

    # Remove the DOCTYPE declaration
    if soup.contents and isinstance(soup.contents[0], Doctype):
        soup.contents[0].extract()

    # Return the modified HTML document
    if pretty:
        return soup.prettify()
    return str(soup)


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Parse an HTML file and rename the <body> tag to <div>"
    )
    parser.add_argument("input_file", help="path to input HTML file", type=pathlib.Path)
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="path to output file (default: output.html)",
    )
    args = parser.parse_args()

    # Parse the input HTML file and write the modified HTML to the output file
    modified_html = parse_html_file(args.input_file)
    if args.output is None:
        args.output = args.input_file.with_suffix(".html").with_stem(
            args.input_file.stem + "_stripped"
        )

    with open(args.output, "w") as file:
        file.write(modified_html)


if __name__ == "__main__":
    main()
