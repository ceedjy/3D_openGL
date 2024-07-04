from OpenGL.GLUT import *
from OpenGL.GL import *
from OpenGL.GL import shaders
import numpy as np
import pywavefront
from time import sleep
from math import cos, sin, tan, pi

import imageio as iio
image = iio.imread('spot_texture.png') #image for the cow
brickwall = iio.imread('brickwall.jpg') #image for the wall
normalMap = iio.imread('brickwall_normal.jpg') #image for the normal map  

scene = pywavefront.Wavefront('spot.obj')

for name, material in scene.materials.items():
    material.vertex_format
    vertices = material.vertices

#-----------------------------------Shader for the cow------------------------------------------------

VERTEX_SHADER = """

#version 330

uniform mat4 camera;   
uniform vec3 camera_pos;
uniform vec3 lumiere;
in vec4 position;
in vec3 normal;
in vec2 TexCoord;
out vec3 n;
out vec4 pos_monde;
out vec3 cam_pos;

out vec2 text_coord;
void main() {
    pos_monde = position;
    vec4 pos = camera * position;
    gl_Position = pos;
    cam_pos = camera_pos;
    n = normal;
    text_coord = TexCoord;
}

"""

FRAGMENT_SHADER = """
#version 330

uniform vec3 lumiere;
in vec3 n;
in vec4 pos_monde;
in vec3 cam_pos;
uniform sampler2D ourTexture;
in vec2 text_coord;
uniform int phong;
void main() {
    float coeff;
    
    // the texture
    vec2 txtCord = vec2(text_coord.x, -(text_coord.y)); // on change le sens de l'image pour qu'elle soit dans le bon sens 
    vec3 couleur_texture = texture(ourTexture, txtCord).xyz;
    
    // diffuse light 
    vec3 vec_light = vec3(lumiere.x - pos_monde.x, lumiere.y - pos_monde.y, lumiere.z - pos_monde.z);
    vec3 norm_light = normalize(vec_light);
    float coeff_coul = dot(n, norm_light);
    float lum_diffuse = max(coeff_coul, 0.0); 
    
    // view vector
    vec3 vec_vue = vec3(cam_pos.x - pos_monde.x, cam_pos.y - pos_monde.y, cam_pos.z - pos_monde.z);
    vec3 norm_vue = normalize(vec_vue);
    
    if (phong == 1){
        // specular light
        vec3 vec_H = vec3(norm_light.x + norm_vue.x, norm_light.y + norm_vue.y, norm_light.z + norm_vue.z);
        vec3 norm_vec_H = normalize(vec_H);
        float alpha = 6.0;
        float spec = max(dot(n, norm_vec_H), 0.0);
        float lum_speculaire = pow(spec, alpha);
        
        // ambient light
        float lum_ambiante = 0.1f; 
        
        // total light coefficient 
        coeff = lum_diffuse + lum_ambiante + lum_speculaire;
        if (coeff > 1.0){
            coeff = 1.0;
        }
    }else{
        if (lum_diffuse > 0.7) {
            coeff = 1.0;
        } else if (lum_diffuse > 0.3){
            coeff = 0.6;
        } else {
            coeff = 0.2;
        }
            
        float contour = dot(n, vec_vue);
        if (contour <= 0.1 && contour >= -0.1){ //contour == 0.0
            coeff = 0.0;
        }
    }
    gl_FragColor = vec4(couleur_texture.r*coeff, couleur_texture.g*coeff, couleur_texture.b*coeff, 1.0f);
}
 
"""

#-----------------------------------Shader for the wall------------------------------------------------

VERTEX_SHADER2 = """

#version 330

uniform mat4 camera2;   
uniform vec3 camera_pos;
uniform vec3 lumiere;
uniform mat4 camera_lum;
in vec4 position;
in vec2 textuCoorMur; 
out vec4 pos_monde;
out vec2 text_coord_mur;
out vec3 cam_pos;
out vec3 trouve_point;
void main() {
    pos_monde = position;
    vec4 pos = camera2 * position;
    gl_Position = pos;
    text_coord_mur = textuCoorMur; 
    cam_pos = camera_pos;
    vec4 trouv_pt = (camera_lum * pos_monde);
    trouve_point = trouv_pt.xyz / trouv_pt.w;
    trouve_point = trouve_point * 0.5 + 0.5;
}

"""

FRAGMENT_SHADER2 = """
#version 330

uniform vec3 lumiere;
in vec4 pos_monde;
in vec3 cam_pos;
uniform sampler2D brickWallText;
uniform sampler2D normalMap;
uniform sampler2D LUMtexture;
in vec2 text_coord_mur;
in vec3 trouve_point;
uniform int phong; 
void main() {
    
    float coeff;
    
    // recovery of normal
    vec3 normal = texture(normalMap, text_coord_mur).rgb;
    normal = normalize(normal * 2.0 - 1.0);  
    normal = vec3(normal.x, normal.y, -(normal.z));
    
    // the texture 
    vec3 couleur_texture = texture(brickWallText, text_coord_mur).xyz;
    
    // diffuse light
    vec3 vec_light = vec3(lumiere.x - pos_monde.x, lumiere.y - pos_monde.y, lumiere.z - pos_monde.z);
    vec3 norm_light = normalize(vec_light);
    float coeff_coul = dot(normal, norm_light);
    float lum_diffuse = max(coeff_coul, 0.0);
    
    // view vector
    vec3 vec_vue = vec3(cam_pos.x - pos_monde.x, cam_pos.y - pos_monde.y, cam_pos.z - pos_monde.z);
    
    // shadow 
    float nearestDepth = texture(LUMtexture, trouve_point.xy).r;
    float currentDepth = trouve_point.z;
    float inShadow = currentDepth < nearestDepth ? 0.0 : 1.0;
    
    if (phong == 1){
        // specular light
        vec3 norm_vue = normalize(vec_vue);
        vec3 vec_H = vec3(norm_light.x + norm_vue.x, norm_light.y + norm_vue.y, norm_light.z + norm_vue.z);
        vec3 norm_vec_H = normalize(vec_H); 
        float alpha = 6.0;
        float spec = max(dot(normal, norm_vec_H), 0.0);
        float lum_speculaire = pow(spec, alpha);
        
        // ambiant light 
        float lum_ambiante = 0.1f; 
        
        // total light coefficient 
        coeff = lum_diffuse + lum_ambiante + lum_speculaire;
        if(inShadow == 1.0){ //si le point est dans l'ombre 
            coeff = lum_ambiante;
        }
        if (coeff > 1.0){
            coeff = 1.0;
        }
    }else{
        if (lum_diffuse > 0.7) {
            coeff = 1.0;
        } else if (lum_diffuse > 0.3){
            coeff = 0.6;
        } else {
            coeff = 0.2;
        }
            
        if(inShadow == 1.0){
            coeff = coeff/2.0;
        }
        
        // add black outline
        float contour = normalize(dot(normal, vec_vue));
        if (contour == 0.0){
            coeff = 0.0;
        }
    }
    gl_FragColor = vec4(couleur_texture.r*coeff, couleur_texture.g*coeff, couleur_texture.b*coeff, 1.0f);
}
 
"""

#--------------------------Shader for the screen of the framebuffer---------------------------

VERTEX_SHADER3 = """

#version 330

in vec4 position;
in vec2 textuCoorFB; 
out vec4 pos;
out vec2 text_coord_fb;
void main() {
    pos = position;
    gl_Position = pos;
    text_coord_fb = textuCoorFB;  
}

"""

FRAGMENT_SHADER3 = """
#version 330

in vec4 pos;
uniform sampler2D FBtexture;
uniform sampler2D ZBFtexture;
uniform vec2 dimension;
in vec2 text_coord_fb;
in vec2 dim;
uniform int filtre;
void main() {
    
    // the texture 
    vec3 couleur_texture = texture(FBtexture, text_coord_fb).xyz;
    
    // les dimensions de la fenetre 
    float W = 1.0/dimension.x;
    float H = 1.0/dimension.y;
    
    // apply the texture to the screen
    // 1: inverting colours 
    // 2: red filter
    // 3: green filter
    // 4: blue filter 
    // 5: yellow filter  
    // 6: slight blur
    // 7: big blur
    
    switch(filtre){
        case 1:
            gl_FragColor = vec4(1-(couleur_texture.r), 1-(couleur_texture.g), 1-(couleur_texture.b), 1.0f); break;
        case 2:
            gl_FragColor = vec4(couleur_texture.r, 0.0, 0.0, 1.0f); break;
        case 3:
            gl_FragColor = vec4(0.0, couleur_texture.g, 0.0, 1.0f); break;
        case 4:
            gl_FragColor = vec4(0.0, 0.0, couleur_texture.b, 1.0f); break;
        case 5:
            gl_FragColor = vec4(couleur_texture.r, couleur_texture.g, 0.0, 1.0f); break;
        case 6:
            // the points to pick up to create the blur (the 8 points around the point being viewed)
            vec3 a = texture(FBtexture, vec2(text_coord_fb.x - W, text_coord_fb.y + H)).xyz;
            vec3 b = texture(FBtexture, vec2(text_coord_fb.x, text_coord_fb.y + H)).xyz;
            vec3 c = texture(FBtexture, vec2(text_coord_fb.x + W, text_coord_fb.y + H)).xyz;
            vec3 d = texture(FBtexture, vec2(text_coord_fb.x - W, text_coord_fb.y)).xyz;
            vec3 e = texture(FBtexture, vec2(text_coord_fb.x + W, text_coord_fb.y)).xyz;
            vec3 f = texture(FBtexture, vec2(text_coord_fb.x - W, text_coord_fb.y - H)).xyz;
            vec3 g = texture(FBtexture, vec2(text_coord_fb.x, text_coord_fb.y - H)).xyz;
            vec3 h = texture(FBtexture, vec2(text_coord_fb.x + W, text_coord_fb.y - H)).xyz;
            
            // the average
            float red = (couleur_texture.r + a.r + b.r + c.r + d.r + e.r + f.r + g.r + h.r)/9.0;
            float green = (couleur_texture.g + a.g + b.g + c.g + d.g + e.g + f.g + g.g + h.g)/9.0;
            float blue = (couleur_texture.b + a.b + b.b + c.b + d.b + e.b + f.b + g.b + h.b)/9.0;
            
            // final color
            gl_FragColor = vec4(red, green, blue, 1.0f); break;
            
         case 7:
            // take the 80 pxels around the point 
            vec3 somme = vec3(0.0, 0.0, 0.0);
            for(int i = -4; i <= 4; i++){
                for(int j = -4; j <= 4; j++){
                    somme += (texture(FBtexture, vec2(text_coord_fb.x + i * W, text_coord_fb.y + j * H)).xyz) / 81.0;
                }
            }
                    
            // final color
            gl_FragColor = vec4(somme.r, somme.g, somme.b, 1.0f); break;
            
        default : // color without change
            gl_FragColor = vec4(couleur_texture.r, couleur_texture.g, couleur_texture.b, 1.0f); break;
    }
}
 
"""

#--------------------------Shader for the shadow map---------------------------

VERTEX_SHADER_LUM = """

#version 330

uniform mat4 camera_lum;
in vec4 position;
void main() {
     gl_Position = camera_lum * position; 
}

"""

FRAGMENT_SHADER_LUM = """
#version 330

void main() {
    // there is nothing to do  
}
 
"""

#-------------------------------------------------------------------------------------------

# global variables :

# perspective matrix : 
# camera viewing angle 
fov = 90.0
# znear (distance from the nearest plane)
n = 0.1
# zfar (distance from the distant plane)
f = 100.0
# visible area 
s = 1./(tan((fov/2.)*(pi/180.)))
# final matrix
matriceP = [[s,0,0,0],
            [0,s,0,0],
            [0,0,-(f+n)/(f-n),-1],
            [0,0,-2*(f*n)/(f-n),0]]

# A global variable containing the program to be run on the graphics card
shaderProgram = None
shaderProgram2 = None
shaderProgram3 = None

# vertex array object 
VAO = None
VAO2 = None
VAO3 = None
VAO_LUM = None

# framebuffer
FBO = None 
FBO_LUM = None 

# textures 
texture = None
texture2 = None
texture3 = None
TEXTURE = None 
TEXTUREzbuf = None 
TEXTURE_LUM = None 

# screen dimensions 
WIDHT = 640
HEIGHT = 480

# global variables that can be changed by pressing keys on the keyboard
# positions 
position_xa = 0.0
position_xb = 0.0
position_xc = 0.0
# the light 
lum_posx = 0.0
lum_posy = 0.0
# camera 
# translation matrix 
Tx = 0.0
Ty = 0.0
Tz = -2.0
# rotation matrix (angle of rotation)
Rx = 0.0
Ry = 180.0
Rz = 0.0
# light camera
# translation matrix
TxL = 0.0
TyL = 0.0
TzL = -2.0
# rotation matrix (angle of rotation)
RxL = 0.0
RyL = 180.0
RzL = 0.0
# phong or toon shading (like a boolean)
phongVache = 1
phongMur = 1
# filter number for the framebuffer
filtre = 0


# FBO screen resizing function based on window dimensions (doesn't work very well)
def redimention(width, height):
    global TEXTURE
    global FBO 
    
    # pick up the new dimensions of the window 
    glViewport(0, 0, width, height)

    # the texture is changed to match the dimensions  
    # open the FBO
    glBindFramebuffer(GL_FRAMEBUFFER, FBO)
    # the texture  
    glBindTexture(GL_TEXTURE_2D, TEXTURE)
    # filling the image with nothing (none)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, None)
    # specify that this texture is linked to the framebuffer and what it is used for 
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, TEXTURE, 0)
    # close the framebuffer 
    glBindFramebuffer(GL_FRAMEBUFFER, 0)


'''
OpenGL initialisation
'''
def initialize():
    global VERTEXT_SHADER
    global FRAGMEN_SHADER
    global shaderProgram
    global VERTEXT_SHADER2
    global FRAGMEN_SHADER2
    global shaderProgram2
    global VERTEXT_SHADER3
    global FRAGMEN_SHADER3
    global shaderProgram3
    global VERTEXT_SHADER_LUM
    global FRAGMEN_SHADER_LUM
    global shaderProgram_lum
    global vertices 
    global texture 
    global texture2 
    global texture3 
    global TEXTURE
    global TEXTUREzbuf
    global TEXTURE_LUM
    global VAO
    global VAO2
    global VAO3
    global VAO_LUM
    global FBO
    global FBO_LUM
    global WIDHT
    global HEIGHT

    # the cow :
    
    # shader for the cow
    vertexshader = shaders.compileShader(VERTEX_SHADER, GL_VERTEX_SHADER)
    fragmentshader = shaders.compileShader(FRAGMENT_SHADER, GL_FRAGMENT_SHADER)
    shaderProgram = shaders.compileProgram(vertexshader, fragmentshader)
    
    # initialise a VAO  
    VAO = glGenVertexArrays(1)

    # the VBO
    vertice = np.array(vertices, dtype=np.float32)
    vertice_VBO = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vertice_VBO)
    glBufferData(GL_ARRAY_BUFFER, vertice.nbytes, vertice, GL_DYNAMIC_DRAW)
    
    # using the VAO
    glBindVertexArray(VAO)

    # normales 
    normal = glGetAttribLocation(shaderProgram, 'normal')
    glVertexAttribPointer(normal, 3, GL_FLOAT, GL_FALSE, 11*4, ctypes.c_void_p(5 * 4))
    glEnableVertexAttribArray(normal)
    # positions 
    position = glGetAttribLocation(shaderProgram, 'position')
    glVertexAttribPointer(position, 3, GL_FLOAT, GL_FALSE, 11*4, ctypes.c_void_p(8 * 4))
    glEnableVertexAttribArray(position)
    # textures 
    TexCoord = glGetAttribLocation(shaderProgram, 'TexCoord')
    glVertexAttribPointer(TexCoord, 2, GL_FLOAT, GL_FALSE, 11*4, None)
    glEnableVertexAttribArray(TexCoord)

    # for the texture 
    texture = glGenTextures(1) 
    glActiveTexture(GL_TEXTURE0)
    glBindTexture(GL_TEXTURE_2D, texture)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    # filling the image 
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 1024, 1024, 0, GL_RGB, GL_UNSIGNED_BYTE, image)
    
    # close the VAO
    glBindVertexArray(0)
    
    # set up the z-buffer
    glEnable(GL_DEPTH_TEST)
    
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    # the wall :
        
    # new shader for the wall
    vertexshader2 = shaders.compileShader(VERTEX_SHADER2, GL_VERTEX_SHADER)
    fragmentshader2 = shaders.compileShader(FRAGMENT_SHADER2, GL_FRAGMENT_SHADER)
    shaderProgram2 = shaders.compileProgram(vertexshader2, fragmentshader2)
    
    VAO2 = glGenVertexArrays(1)
    
    # the VBO 
    #       #triangle 1       #texture 1
    infos = [-1.0, -1.0, 1.0, 0.0, 0.0, 
             -1.0, 1.0, 1.0, 0.0, 1.0,
             1.0, -1.0, 1.0, 1.0, 0.0,
             #triangle 2      #texture 2
             -1.0, 1.0, 1.0, 0.0, 1.0,
             1.0, -1.0, 1.0, 1.0, 0.0,
             1.0, 1.0, 1.0, 1.0, 1.0]
             
    infos = np.array(infos, dtype=np.float32)
    infos_VBO = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, infos_VBO)
    glBufferData(GL_ARRAY_BUFFER, infos.nbytes, infos, GL_DYNAMIC_DRAW)
    
    # initialise a VAO
    glBindVertexArray(VAO2)
    
    # positions  
    position = glGetAttribLocation(shaderProgram2, 'position')
    glVertexAttribPointer(position, 3, GL_FLOAT, GL_FALSE, 5*4, None)
    glEnableVertexAttribArray(position)
    # textures 
    textuCoorMur = glGetAttribLocation(shaderProgram2, 'textuCoorMur')
    glVertexAttribPointer(textuCoorMur, 2, GL_FLOAT, GL_FALSE, 5*4, ctypes.c_void_p(3 * 4))
    glEnableVertexAttribArray(textuCoorMur)
    
    # for the texture of the wall
    texture2 = glGenTextures(1) 
    glActiveTexture(GL_TEXTURE1)
    glBindTexture(GL_TEXTURE_2D, texture2)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    # filling the image
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 1024, 1024, 0, GL_RGB, GL_UNSIGNED_BYTE, brickwall)
    
    # for the texture of the normal map 
    texture3 = glGenTextures(1) 
    glActiveTexture(GL_TEXTURE2)
    glBindTexture(GL_TEXTURE_2D, texture3)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    # filling the image
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 1024, 1024, 0, GL_RGB, GL_UNSIGNED_BYTE, normalMap)
    
    # close the VAO
    glBindVertexArray(0)
    
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # the framebuffer 
    
    # new shader for the screen   
    vertexshader3 = shaders.compileShader(VERTEX_SHADER3, GL_VERTEX_SHADER)
    fragmentshader3 = shaders.compileShader(FRAGMENT_SHADER3, GL_FRAGMENT_SHADER)
    shaderProgram3 = shaders.compileProgram(vertexshader3, fragmentshader3)
    
    VAO3 = glGenVertexArrays(1)
    
    # the VBO 
    #       #triangle 1       #texture 1
    infos = [-1.0, -1.0, 0.0, 0.0, 0.0, 
             -1.0, 1.0, 0.0, 0.0, 1.0,
             1.0, -1.0, 0.0, 1.0, 0.0,
             #triangle 2      #texture 2
             -1.0, 1.0, 0.0, 0.0, 1.0,
             1.0, -1.0, 0.0, 1.0, 0.0,
             1.0, 1.0, 0.0, 1.0, 1.0]
             
    infosFB = np.array(infos, dtype=np.float32)
    infos_VBO3 = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, infos_VBO3)
    glBufferData(GL_ARRAY_BUFFER, infosFB.nbytes, infosFB, GL_DYNAMIC_DRAW)
    
    # initialise a VAO
    glBindVertexArray(VAO3)
    
    # positions  
    position = glGetAttribLocation(shaderProgram3, 'position')
    glVertexAttribPointer(position, 3, GL_FLOAT, GL_FALSE, 5*4, None)
    glEnableVertexAttribArray(position)
    # texture 
    textuCoorFB = glGetAttribLocation(shaderProgram3, 'textuCoorFB')
    glVertexAttribPointer(textuCoorFB, 2, GL_FLOAT, GL_FALSE, 5*4, ctypes.c_void_p(3 * 4))
    glEnableVertexAttribArray(textuCoorFB)
    
    # close the VAO
    glBindVertexArray(0)
    
    # open FBO
    FBO = glGenFramebuffers(1)
    glBindFramebuffer(GL_FRAMEBUFFER, FBO)
    
    # open texture for the color
    TEXTURE = glGenTextures(1)
    glActiveTexture(GL_TEXTURE3) 
    glBindTexture(GL_TEXTURE_2D, TEXTURE)
    # filling the image with nothing (none)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, WIDHT, HEIGHT, 0, GL_RGB, GL_UNSIGNED_BYTE, None)
    # add parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    # specify that this texture is linked to the framebuffer and what it is used for 
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, TEXTURE, 0)

    # open texture for the le z-buffer 
    TEXTUREzbuf = glGenTextures(1)
    glActiveTexture(GL_TEXTURE4) 
    glBindTexture(GL_TEXTURE_2D, TEXTUREzbuf)
    # filling the image with nothing (none)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT32F, WIDHT, HEIGHT, 0, GL_DEPTH_COMPONENT, GL_FLOAT, None)
    # add parameters 
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    # specify that this texture is linked to the framebuffer and what it is used for 
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, TEXTUREzbuf, 0)
    
    # close the framebuffer 
    glBindFramebuffer(GL_FRAMEBUFFER, 0)

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # the framebuffer for the light
    
    # new shader for the image of light
    vertexshader_lum = shaders.compileShader(VERTEX_SHADER_LUM, GL_VERTEX_SHADER)
    fragmentshader_lum = shaders.compileShader(FRAGMENT_SHADER_LUM, GL_FRAGMENT_SHADER)
    shaderProgram_lum = shaders.compileProgram(vertexshader_lum, fragmentshader_lum)
    
    # initialise a VAO  
    VAO_LUM = glGenVertexArrays(1)

    # the VBO
    vertices_lum = np.array(vertices, dtype=np.float32)
    vertices_VBO_LUM = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vertices_VBO_LUM)
    glBufferData(GL_ARRAY_BUFFER, vertices_lum.nbytes, vertices_lum, GL_DYNAMIC_DRAW)
    
    # using thee VAO
    glBindVertexArray(VAO_LUM)

    # positions 
    position = glGetAttribLocation(shaderProgram_lum, 'position')
    glVertexAttribPointer(position, 3, GL_FLOAT, GL_FALSE, 11*4, ctypes.c_void_p(8 * 4))
    glEnableVertexAttribArray(position)

    # close the VAO
    glBindVertexArray(0)
    
    # open the FBO
    FBO_LUM = glGenFramebuffers(1)
    glBindFramebuffer(GL_FRAMEBUFFER, FBO_LUM)

    # open texture for the z-buffer 
    TEXTURE_LUM = glGenTextures(1)
    glActiveTexture(GL_TEXTURE5) 
    glBindTexture(GL_TEXTURE_2D, TEXTURE_LUM)
    # filling the image with northing (none)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT32F, WIDHT, HEIGHT, 0, GL_DEPTH_COMPONENT, GL_FLOAT, None)
    # add parameters 
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    # specify that this texture is linked to the framebuffer and what it is used for 
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, TEXTURE_LUM, 0)
    
    # close the framebuffer 
    glBindFramebuffer(GL_FRAMEBUFFER, 0)
    

'''
Function called each time a new image is show
'''
def render():
    global shaderProgram
    global shaderProgram2
    global shaderProgram3
    global shaderProgram_lum
    global VAO
    global VAO2
    global VAO3
    global VAO_LUM
    global FBO
    global FBO_LUM
    global lum_posx
    global lum_posy
    global vertices 
    global matriceP
    global TEXTURE_LUM
    global Tx 
    global Ty 
    global Tz 
    global Rx 
    global Ry 
    global Rz 
    global TxL
    global TyL 
    global TzL 
    global RxL 
    global RyL 
    global RzL
    global WIDHT
    global HEIGHT 
    global phongVache
    global phongMur
    global filtre 
    
    # add a sleep to avoid displaying the window 1000 times a second 
    sleep(0.01)
    
    # grey backgroung to see what is in black
    glClearColor(0.1, 0.1, 0.1, 1) 
    
    # clear buffers 
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    # the camera - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

    # transformation matrix :
        
    # dilatation matrix (put in for the theory but in practice it is not used for the camera): 
    dil = [[1, 0, 0, 0], 
           [0, 1, 0, 0],
           [0, 0, 1, 0],
           [0, 0, 0, 1]]

    # translation matrix : 
    tra = [[1, 0, 0, 0], 
           [0, 1, 0, 0],
           [0, 0, 1, 0],
           [Tx, Ty, Tz, 1]]
        
    # rotation matrix :
    # rotation x axis : 
    # rotation angle 
    a = (Rx*pi)/180 
    c = cos(a)
    s = sin(a)
    # matrix for x axis
    rot_x = [[1,0,0,0],
             [0,c,-s,0],
             [0,s,c,0],
             [0,0,0,1]]
    # rotation y axis : 
    # rotation angle
    a = (Ry*pi)/180 
    c = cos(a)
    s = sin(a)
    # matrice for y axis
    rot_y = [[c,0,s,0],
             [0,1,0,0],
             [-s,0,c,0],
             [0,0,0,1]]
    # rotation z axis : 
    # rotation axis
    a = (Rz*pi)/180 
    c = cos(a)
    s = sin(a)
    # matrix for z axis
    rot_z = [[c,-s,0,0],
             [s,c,0,0],
             [0,0,1,0],
             [0,0,0,1]]
    # final matrix : 
    rot = np.dot(rot_x,rot_y)
    rot = np.dot(rot, rot_z)
    
    # transformation matrix complete : 
    matriceT = np.dot(dil, rot)
    matriceT = np.dot(matriceT, tra)
    matriceT = np.linalg.inv(matriceT)
    
    
    # camera matrix with perspective and transformations :
    cameraTP = np.dot(matriceT, matriceP)
    
    # the light camera - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    
    # transformation matrix :
        
    # translation matrix : 
    tra_lum = [[1, 0, 0, 0], 
           [0, 1, 0, 0],
           [0, 0, 1, 0],
           [TxL, TyL, TzL, 1]]
        
    # rotation matrix :
    # rotation x axis : 
    # rotation angle
    a = (RxL*pi)/180 
    c = cos(a)
    s = sin(a)
    # matrice for x axis
    rot_x_lum = [[1,0,0,0],
             [0,c,-s,0],
             [0,s,c,0],
             [0,0,0,1]]
    # rotation y axis : 
    # rotation angle
    a = (RyL*pi)/180 
    c = cos(a)
    s = sin(a)
    # matrice for y axis
    rot_y_lum = [[c,0,s,0],
             [0,1,0,0],
             [-s,0,c,0],
             [0,0,0,1]]
    # rotation z axis : 
    # rotation angle
    a = (RzL*pi)/180 
    c = cos(a)
    s = sin(a)
    # matrice for z axis
    rot_z_lum = [[c,-s,0,0],
             [s,c,0,0],
             [0,0,1,0],
             [0,0,0,1]]
    # final matrix : 
    rot_lum = np.dot(rot_x_lum,rot_y_lum)
    rot_lum = np.dot(rot_lum, rot_z_lum)
    
    # transformation matrix complete : 
    matriceT_lum = np.dot(dil, rot_lum)
    matriceT_lum = np.dot(matriceT_lum, tra_lum)
    matriceT_lum = np.linalg.inv(matriceT_lum)
    
    # light camera matrix with perspective and transformations : 
    camera_lum = np.dot(matriceT_lum, matriceP)

    # shadow map : - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    
    # bind the framebufferfor for the light to draw the rendering image 
    glBindFramebuffer(GL_FRAMEBUFFER, FBO_LUM)
    # clear the framebuffer for the light
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    
    # using new shader
    glUseProgram(shaderProgram_lum)
    # reusing VAO 
    glBindVertexArray(VAO_LUM)
    
    # put uniform :
    # light camera (transformations) :
    location_lum = glGetUniformLocation(shaderProgram_lum, 'camera_lum')
    glUniformMatrix4fv(location_lum, 1, GL_FALSE, camera_lum)
    
    # draw the object : 
    glDrawArrays(GL_TRIANGLES, 0, 5856*3)
    
    # close program
    glUseProgram(0)
    
    # close framebuffer for the light
    glBindFramebuffer(GL_FRAMEBUFFER, 0)
    
    # the framebuffer - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

    # bind the framebuffer to draw the rendering image
    glBindFramebuffer(GL_FRAMEBUFFER, FBO)
    # clear framebuffer 
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    
    # the cow : - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    # using corresponding shader
    glUseProgram(shaderProgram)
    # reusing VAO 
    glBindVertexArray(VAO)
    
    # put uniform :
    # texture for the cow
    glBindTexture(GL_TEXTURE_2D, texture)
    textureLocation = glGetUniformLocation(shaderProgram, "ourTexture")
    glUniform1i(textureLocation, 0) 
    # light 
    lumiere = glGetUniformLocation(shaderProgram, 'lumiere')
    glUniform3f(lumiere, lum_posx, lum_posy, -1.0) 
    # camera (position)
    camera_pos = glGetUniformLocation(shaderProgram, 'camera_pos')
    glUniform3f(camera_pos, Tx, Ty, Tz)
    
    # add camera (transformations) :
    location = glGetUniformLocation(shaderProgram, 'camera')
    glUniformMatrix4fv(location, 1, GL_FALSE, cameraTP)
    
    # booleen for the illumination of blinn phong 
    phong = glGetUniformLocation(shaderProgram, 'phong')
    glUniform1i(phong, phongVache)
    
    # draw the object : 
    glDrawArrays(GL_TRIANGLES, 0, 5856*3)
    
    # close program
    glUseProgram(0)
    
    # the wall : - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
    
    # using new shader for the wall 
    glUseProgram(shaderProgram2)
    #on reutilise le VAO2
    glBindVertexArray(VAO2) 
    
    # put uniform :
    # texture for the wall 
    glBindTexture(GL_TEXTURE_2D, texture2)
    textureLocation = glGetUniformLocation(shaderProgram2, 'brickWallText')
    glUniform1i(textureLocation, 1)
    # texture normal map 
    glBindTexture(GL_TEXTURE_2D, texture3)
    textureLocationNormal = glGetUniformLocation(shaderProgram2, 'normalMap') 
    glUniform1i(textureLocationNormal, 2)
    # zbuffer for the light (texture for z-buffer)
    glBindTexture(GL_TEXTURE_2D, TEXTURE_LUM)
    textureLocationLUM = glGetUniformLocation(shaderProgram2, 'LUMtexture')
    glUniform1i(textureLocationLUM, 5) 
    # light
    lumiere = glGetUniformLocation(shaderProgram2, 'lumiere')
    glUniform3f(lumiere, lum_posx, lum_posy, -1.0) 
    # camera (position)
    camera_pos = glGetUniformLocation(shaderProgram2, 'camera_pos')
    glUniform3f(camera_pos, Tx, Ty, Tz)
    
    # add camera (transformations) :
    location2 = glGetUniformLocation(shaderProgram2, 'camera2')
    glUniformMatrix4fv(location2, 1, GL_FALSE, cameraTP)
    
    # light camera (transformations) :
    location_lum = glGetUniformLocation(shaderProgram2, 'camera_lum')
    glUniformMatrix4fv(location_lum, 1, GL_FALSE, camera_lum)
    
    # booleen for the illumination of blinn phong 
    phong = glGetUniformLocation(shaderProgram2, 'phong')
    glUniform1i(phong, phongMur)
    
    # draw the object : 
    glDrawArrays(GL_TRIANGLES, 0, 2*3)
    
    # close program
    glUseProgram(0)
    
    # close framebuffer 
    glBindFramebuffer(GL_FRAMEBUFFER, 0)
    
    # screen for the framebuffer : - - - - - - - - - - - - - - - - - - - - - - 
    
    # new shader 
    glUseProgram(shaderProgram3)
    #on reutilise le VAO3
    glBindVertexArray(VAO3) 
    
    # put uniform :
    # texture for the screen 
    glBindTexture(GL_TEXTURE_2D, TEXTURE)
    textureLocationFB = glGetUniformLocation(shaderProgram3, 'FBtexture')
    glUniform1i(textureLocationFB, 3) 
    
    # zbuffer for the scren 
    glBindTexture(GL_TEXTURE_2D, TEXTUREzbuf)
    textureLocationZBF = glGetUniformLocation(shaderProgram3, 'ZBFtexture')
    glUniform1i(textureLocationZBF, 4)
    # dimensions of the window
    dimension = glGetUniformLocation(shaderProgram3, 'dimension')
    glUniform2f(dimension, WIDHT, HEIGHT)
    
    # filter number
    noFiltre = glGetUniformLocation(shaderProgram3, 'filtre')
    glUniform1i(noFiltre, filtre)
    
    # draw the object : 
    glDrawArrays(GL_TRIANGLES, 0, 2*3)
    
    # close programme
    glUseProgram(0)


    # update image then redraw  
    glutSwapBuffers()
    glutPostRedisplay()
    

'''
Function called each time a key is pressed
'''
def keyboard(key,x,y):
    global position_xa
    global position_xb
    global position_xc
    global lum_posx
    global lum_posy
    global Tx
    global Ty 
    global Tz 
    global Rx 
    global Ry 
    global Rz 
    global TxL
    global TyL 
    global TzL 
    global RxL 
    global RyL 
    global RzL
    global phongVache
    global phongMur
    global filtre 
        
    # light x axis
    if key == b'm':
        lum_posx -= 0.1
        TxL -= 0.1
    if key == b'k':
        lum_posx += 0.1
        TxL += 0.1
    # light y axis
    if key == b'o':
        lum_posy += 0.1
        TyL += 0.1
    if key == b'l':
        lum_posy -= 0.1
        TyL -= 0.1
    
    # translation : 
    if key == b'q':
        Ry_bis = ((Ry-90)*pi)/180 
        Tz -= cos(Ry_bis)*0.1
        Tx += sin(Ry_bis)*0.1
    if key == b'd':
        Ry_bis = ((Ry+90)*pi)/180 
        Tz -= cos(Ry_bis)*0.1
        Tx += sin(Ry_bis)*0.1
    if key == b'a':
        Ty += 0.1
    if key == b'e':
        Ty -= 0.1
    if key == b'z':
        Ry_bis = (Ry*pi)/180 
        Tz -= cos(Ry_bis)*0.1
        Tx += sin(Ry_bis)*0.1
    if key == b's':
        Ry_bis = (Ry*pi)/180 
        Tz += cos(Ry_bis)*0.1
        Tx -= sin(Ry_bis)*0.1
        
    # rotation : 
    if key == b'g':
        Rx += 10.0
    if key == b't':
        Rx -= 10.0
    if key == b'h': 
        Ry += 10.0
    if key == b'f':
        Ry -= 10.0
    if key == b'r':
        Rz += 10.0
    if key == b'y':
        Rz -= 10.0
        
    # blinn phong illumination and toon shading
    if key == b'b':
        if phongVache == 1:
            phongVache = 0
        else: 
            phongVache = 1
    if key == b'n':
        if phongMur == 1:
            phongMur = 0
        else: 
            phongMur = 1
            
    # filter 
    if key == b'v':
        filtre += 1
        if filtre > 7:
            filtre = 0
        
        
'''
GLUT window
'''
def main():

    glutInit([])
    glutInitWindowSize(640, 480)
    glutCreateWindow("pyopengl with glut")
    initialize()
    glutDisplayFunc(render)
    glutReshapeFunc(redimention)
    glutKeyboardFunc(keyboard)
    glutMainLoop()


if __name__ == '__main__':
    main()
