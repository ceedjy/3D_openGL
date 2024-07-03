# Presentation 
An experimental course project for 3D model rendering with openGL in Python and GLSL. 

# Informations
A project as part of the VISI401 course at Savoie Mont Blanc University in 2024. \
Tutor : Colin Weill-Duflos 

# Run the code 
You need to have OpenGL installed, and more precisely GLUT, to run the code. \
Then run the code in the file 3D_opengl.py and enjoy the project. 

# Real-time rendering 
You have several controls over this 3D model, such as : 
- The position of the light :
  - x axis : 'm' + & 'k' -
  - y axis : 'o' + & 'l' -
- The position of the camera :
  - translation : 
    - left : 'q' 
    - right : 'd' 
    - top : 'a'
    - bottom : 'e'
    - front : 'z'
    - back : 's'
  - rotation : 
    - left : 'f'
    - right : 'h'
    - up : 't'
    - down : 'g'
    - lean left : 'r'
    - lean right : 'y'
- Change the illumination model (Phong Shading modele - Toon Shading) :
    - toggle the cow : 'b'
    - toggle the wall : 'n'
- Change the filter : 'v'. Each click will change the filter in this order :
    - no filter
    - inverting colours
    - red filter
    - green filter
    - blue filter
    - yellow filter
    - slight blur
    - big blur 
