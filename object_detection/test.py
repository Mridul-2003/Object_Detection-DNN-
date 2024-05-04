import requests

# URL of your Flask API endpoint
url = 'http://localhost:5000/detect_objects'  # Replace with your actual endpoint URL

# Path to the image file
image_path = '/Users/useradmin/Documents/object_detection/soccer.jpg'


# Open the image file in binary mode
with open(image_path, 'rb') as img_file:
    # Send a POST request with the image file
    response = requests.post(url, files={'image': img_file})

# Check the response
if response.status_code == 200:
    # If the request was successful, save the image file
    with open('detected_image.jpg', 'wb') as output_file:
        output_file.write(response.content)
    print('Image with detected objects saved as detected_image.jpg')
else:
    # If there was an error, print the error message
    print('Error:', response.text)
