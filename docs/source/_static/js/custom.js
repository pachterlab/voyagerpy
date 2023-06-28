document.addEventListener("DOMContentLoaded", function () {
	// Get all elements with the class name "rendered_html"
	var elements = document.getElementsByClassName("rendered_html");

	// Convert the HTMLCollection to an array for easier manipulation
	var elementsArray = Array.from(elements);

	// Remove the class "rendered_html" from each element
	elementsArray.forEach(function (element) {
		element.classList.remove("rendered_html");
	});
});