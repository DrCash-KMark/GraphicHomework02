//=============================================================================================
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat, BOM kihuzando.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiveve
// - Mashonnan atvett programresszleteket forrasmegjeloles nelkul felhasznalni es
// - felesleges programsorokat a beadott programban hagyni!!!!!!!
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak
// A keretben nem szereplo GLUT fuggvenyek tiltottak.
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : Karpati Mark Andras
// Neptun : O1BG0Z
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================
#include "framework.h"

const vec3 N = vec3(0.17, 0.35, 1.5);
const vec3 KAPPA = vec3(3.1, 2.7, 1.9);
//const vec3 F0 = ((N - 1.0f) * (N - 1.0f) + K * K) / ((N + 1.0f) * (N + 1.0f) + K * K);
const float epsilon = 0.0001f;
const vec3 one(1, 1, 1);
//do i need this?
const float scale = 1.0f;

float rnd() { return (float) rand() / RAND_MAX; }

enum MaterialType {
    ROUGH, REFLECTIVE
};

struct Material {
    vec3 ka, kd, ks, F0;
    float shininess;
    MaterialType type;

    Material(MaterialType t) { type = t; }
};

struct RoughMaterial : Material {
    RoughMaterial(vec3 _kd, vec3 _ks, float _shininess) : Material(ROUGH) {
        ka = _kd * M_PI;
        kd = _kd;
        ks = _ks;
        shininess = _shininess;
    }
};

vec3 operator/(vec3 num, vec3 denom) {
    return vec3(num.x / denom.x, num.y / denom.y, num.z / denom.z);
}

struct ReflectiveMaterial : Material {
    ReflectiveMaterial(vec3 n, vec3 kappa) : Material(REFLECTIVE) {
        F0 = ((n - one) * (n - one) + kappa * kappa) / ((n + one) * (n + one) + kappa * kappa);
    }
};

struct Hit {
    float t;
    vec3 position, normal;
    Material *material;

    Hit() { t = -1; }
};

struct Ray {
    vec3 start, dir;

    Ray(vec3 _start, vec3 _dir) {
        start = _start;
        dir = normalize(_dir);
    }
};

vec3 fresnel(vec3 F0, vec3 v, vec3 n) {
    float cosTheata = dot(-v, n);
    vec3 one(1, 1, 1);
    vec3 returnValue = F0 + (one - F0) * (pow(1 - cosTheata, 5));
    return returnValue;
}

class Intersectable {
protected:
    Material *material;
public:
    virtual Hit intersect(const Ray &ray) = 0;
};

struct Sphere : public Intersectable {
    vec3 center;
    float radius;
public:
    Sphere(const vec3 &_center, float _radius, Material *_material) {
        center = _center;
        radius = _radius;
        material = _material;
    }

    Hit intersect(const Ray &ray) {
        Hit hit;
        vec3 dist = ray.start - center;
        float a = dot(ray.dir, ray.dir);
        float b = dot(dist, ray.dir) * 2.0f;
        float c = dot(dist, dist) - radius * radius;
        float discr = b * b - 4.0f * a * c;
        if (discr < 0) return hit;
        float sqrt_discr = sqrtf(discr);
        float t1 = (-b + sqrt_discr) / 2.0f / a;    // t1 >= t2 for sure
        float t2 = (-b - sqrt_discr) / 2.0f / a;
        if (t1 <= 0) return hit;
        hit.t = (t2 > 0) ? t2 : t1;
        hit.position = ray.start + ray.dir * hit.t;
        hit.normal = (hit.position - center) * (1.0f / radius);
        hit.material = material;
        return hit;
    }
};

struct Quadrics : public Intersectable {
    mat4 Q;
    vec3 pointOfSphare;
    float radius;
    vec3 translation;

    Quadrics(mat4 &_Q, vec3 _pointOfSphare, float _radius, vec3 _translation, Material *_material) {
        Q = _Q;
        pointOfSphare = _pointOfSphare;
        radius = _radius;
        translation = _translation;
        material = _material;
    }

    vec3 gradf(vec3 r) {
        vec4 g = vec4(r.x, r.y, r.z, 1) * Q * 2;
        return vec3(g.x, g.y, g.z);
    }

    Hit intersect(const Ray &ray) {
        Hit hit;
        vec3 start = ray.start - translation;
        vec4 S(start.x, start.y, start.z, 1), D(ray.dir.x, ray.dir.y, ray.dir.z, 0);
        float a = dot(D * Q, D), b = dot(S * Q, D) * 2, c = dot(S * Q, S);
        float discr = b * b - 4.0f * a * c;
        if (discr < 0) return hit;
        float sqrt_discr = sqrtf(discr);

        float t1 = (-b + sqrt_discr) / 2.0f / a;
        vec3 p1 = ray.start + ray.dir * t1;
        float dp1 = sqrtf(powf(pointOfSphare.x - p1.x, 2) +
                          powf(pointOfSphare.y - p1.y, 2) +
                          powf(pointOfSphare.z - p1.z, 2)); //distance of point1
        if (dp1 > radius) {
            t1 = -1;
        }

        float t2 = (-b - sqrt_discr) / 2.0f / a;
        vec3 p2 = ray.start + ray.dir * t2;
        float dp2 = sqrtf(powf(pointOfSphare.x - p2.x, 2) +
                          powf(pointOfSphare.y - p2.y, 2) +
                          powf(pointOfSphare.z - p2.z, 2)); //distance of point2
        if (dp2 > radius) {
            t2 = -1;
        }

        if (t1 <= 0 && t2 <= 0) {
            return hit;
        }
        else if (t1 <= 0) {
            hit.t = t2;
        }
        else if (t2 <= 0) {
            hit.t = t1;
        }
        else if (t2 < t1) {
            hit.t = t2;
        }
        else {
            hit.t = t1;
        }

        hit.position = start + ray.dir * hit.t;
        hit.normal = normalize(gradf(hit.position));
        hit.position = hit.position + translation;
        hit.material = material;

        if (dot(hit.normal, ray.dir) >= 0) {
            hit.normal = hit.normal * -1;
        }
        hit.material = material;

        return hit;
    }


};


struct ConvexPolyhedron : public Intersectable {
    //TODO
    static const int objFaces = 12;
    /*vec3 v[20];
    int planes[objFaces * 3];//12*5*/
    std::vector<vec3> v;
    std::vector<int> planes;
    //Material portal;
public:
    ConvexPolyhedron() {
        //no better idea in the night
        v = {vec3(0, 0.618, 1.618), vec3(0, -0.618, 1.618), vec3(0, -0.618, -1.618),
             vec3(0, 0.618, -1.618), vec3(1.618, 0, 0.618), vec3(-1.618, 0, 0.618),
             vec3(-1.618, 0, -0.618), vec3(1.618, 0, -0.618), vec3(0.618, 1.618, 0),
             vec3(-0.618, 1.618, 0), vec3(-0.618, -1.618, 0), vec3(0.618, -1.618, 0),
             vec3(1, 1, 1), vec3(-1, 1, 1), vec3(-1, -1, 1), vec3(1, -1, 1),
             vec3(1, -1, -1), vec3(1, 1, -1), vec3(-1, 1, -1), vec3(-1, -1, -1)};
        planes = {
                1, 2, 16, 5, 13,    1, 13, 9, 10, 14,   1, 14, 6, 15, 2,    2, 15, 11, 12, 16,
                3, 4, 18, 8, 17,    3, 17, 12, 11, 20,  3, 20, 7, 19, 4,    19, 10, 9, 18, 4,
                16, 12, 17, 8, 5,    5, 8, 18, 9, 13,   14, 10, 19, 7, 6,    6, 7, 20, 11, 15
        };
        material=new RoughMaterial(vec3(0.3f, 0.2f, 0.1f), vec3(2, 2, 2),50);
    }

    void getObjPlane(int i, float scale, vec3 p, vec3 normal) {
        vec3 p1 = v[planes[3 * i] - 1], p2 = v[planes[3 * i + 1] - 1], p3 = v[planes[3 * i + 2] - 1];
        normal = cross(p2 - p1, p3 - p1);
        if (dot(p1, normal) < 0) {
            normal = -normal;
        }
        p = p1 * scale + vec3(0, 0, 0);//we do not need to translat it
    }


    Hit intersect(const Ray& ray) {
        Hit hit;
        for (int i = 0; i < 12; i++) {

            vec3 p1 = v[planes[5 * i] - 1], p2 = v[planes[5 * i + 1] - 1], p3 = v[planes[5 * i + 2] - 1];
            vec3 normal = cross(p2 - p1, p3 - p1);
            if (dot(p1, normal) < 0) normal = -normal;
            vec3 point = p1;
            float t = abs(dot(normal, ray.dir))>epsilon? dot(point - ray.start, normal)/ dot(normal, ray.dir): -1;
            if (t>epsilon && (hit.t>t || hit.t<=0)) {
                vec3 intersectPoint = ray.start + ray.dir * t;
                bool inside = true;
                for (int j = 0; j < 12; j++) {
                    if (j != i) {
                        p1 = v[planes[5 * j] - 1]; p2 = v[planes[5 * j + 1] - 1]; p3 = v[planes[5 * j + 2] - 1];
                        vec3 normalOther = cross(p2 - p1, p3 - p1);
                        if (dot(normalOther, intersectPoint - p1) > 0) {
                            inside = false;
                            break;
                        }

                    }
                }
                if (inside) {
                    hit.t = t;
                    hit.normal = normalize(normal);
                    hit.position = intersectPoint;
                    hit.material = material;
                }
            }
        }
        return hit;
    }
    /*Hit intersect(const Ray &ray) {
        Hit hit;
        for (int i = 0; i < objFaces; ++i) {
            vec3 p1, normal;
            getObjPlane(i, scale, p1, normal);
            float ti = abs(dot(normal, ray.dir)) > epsilon ? dot(p1 - ray.start, normal) / dot(normal, ray.dir) : -1;
            if (ti <= epsilon || (ti > hit.t && hit.t > 0)) {
                continue;
            }
            vec3 printersect = ray.start + ray.dir * ti;
            bool outside = false;
            for (int j = 0; j < objFaces || !outside; ++j) {
                if (i == j) {
                    continue;
                }
                vec3 p11, n;
                getObjPlane(j, scale, p11, n);
                if (dot(n, printersect - p11) > 0) {
                    outside = true;
                }
            }
            if (!outside) {
                hit.t = ti;
                hit.position = printersect;
            }
        }
        return hit;
    }*/

};

class Camera {
    vec3 eye, lookat, right, up;
    float fov;
public:
    void set(vec3 _eye, vec3 _lookat, vec3 vup, float _fov) {
        eye = _eye;
        lookat = _lookat;
        fov = _fov;
        vec3 w = eye - lookat;
        float windowsSize = length(w) * tanf(fov / 2);
        right = normalize(cross(vup, w)) * windowsSize;
        up = normalize(cross(w, right)) * windowsSize;
    }

    Ray getRay(int X, int Y) {
        vec3 dir =
                lookat + right * (2.0f * (X + 0.5f) / windowWidth - 1) + up * (2.0f * (Y + 0.5f) / windowHeight - 1) -
                eye;
        return Ray(eye, dir);
    }

    void Animate(float dt) {
        vec3 d = eye - lookat;
        eye = vec3(d.x * cosf(dt) + d.z * sinf(dt), d.y, -d.x * sinf(dt) + d.z * cosf(dt)) + lookat;
        set(eye, lookat, up, fov);
    }

};

struct Light {
    vec3 direction;
    vec3 Le;

    Light(vec3 _direction, vec3 _Le) {
        direction = normalize(_direction);
        Le = _Le;
    }
};


class Scene {
    std::vector<Intersectable *> objects;
    std::vector<Light *> lights;
    Camera camera;
    vec3 La; //ambient light
public:
    void build() {
        vec3 eye = vec3(5, 0, 1), vup = vec3(0, 1, 0), lookat = vec3(0, 0, 0);
        float fov = 45 * M_PI / 180;
        camera.set(eye, lookat, vup, fov);

        La = vec3(0.4f, 0.4f, 0.4f);
        vec3 lightDirection(1, 1, 1), Le(2, 2, 2);
        lights.push_back(new Light(lightDirection, Le));

        vec3 kd(0.3f, 0.2f, 0.1f), ks(2, 2, 2);
        Material *material01 = new RoughMaterial(kd, ks, 50);
        Material *material02 = new ReflectiveMaterial(N, KAPPA);

        /*
         * need to learn how to make parameters
         */
        float a = 1.5, b = 2.5, c = 0.5;
        mat4 paraboloid = mat4(a, 0, 0, 0,
                               0, b, 0, 0,
                               0, 0, 0, -c,
                               0, 0, -c, 0);
        objects.push_back(new Quadrics(paraboloid, vec3(0.0, 0.0, 0.0), 0.3, vec3(0.0f, 0.0f, 0.0f), material02));

        //objects.push_back(new Sphere(vec3(0.2f, 0.1f, 0.4f), 0.1f, material01));
        //objects.push_back(new Sphere(vec3(-0.1f, -0.2f, -0.3f), 0.1f, material01));
        objects.push_back(new ConvexPolyhedron());
    }

    void render(std::vector<vec4> &image) {
        for (int Y = 0; Y < windowHeight; Y++) {
#pragma omp parallel for
            for (int X = 0; X < windowWidth; X++) {
                vec3 color = trace(camera.getRay(X, Y));
                image[Y * windowWidth + X] = vec4(color.x, color.y, color.z, 1);
            }
        }
    }

    Hit firstIntersect(Ray ray) {
        Hit bestHit;
        for (Intersectable *object : objects) {
            Hit hit = object->intersect(ray); //  hit.t < 0 if no intersection
            if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t)) {
                bestHit = hit;
            }
        }
        if (dot(ray.dir, bestHit.normal) > 0) {
            bestHit.normal = bestHit.normal * (-1);
        }

        return bestHit;
    }

    bool shadowIntersect(Ray ray) {    // for directional lights
        for (Intersectable *object : objects) if (object->intersect(ray).t > 0) return true;
        return false;
    }

    vec3 trace(Ray ray, int depth = 0) {
        if (depth > 5) return La;
        Hit hit = firstIntersect(ray);
        if (hit.t < 0) return La;

        vec3 outRadiance(0, 0, 0);
        if (hit.material->type == ROUGH) {
            outRadiance = hit.material->ka * La;
            for (Light *light : lights) {
                Ray shadowRay(hit.position + hit.normal * epsilon, light->direction);
                float cosTheta = dot(hit.normal, light->direction);
                if (cosTheta > 0 && !shadowIntersect(shadowRay)) {    // shadow computation
                    outRadiance = outRadiance + light->Le * hit.material->kd * cosTheta;
                    vec3 halfway = normalize(-ray.dir + light->direction);
                    float cosDelta = dot(hit.normal, halfway);
                    if (cosDelta > 0) {
                        outRadiance =
                                outRadiance + light->Le * hit.material->ks * powf(cosDelta, hit.material->shininess);
                    }

                }
            }
        }
        if (hit.material->type == REFLECTIVE) {
            vec3 reflectedDir = ray.dir - hit.normal * dot(hit.normal, ray.dir) * 2.0f;
            vec3 F = fresnel(hit.material->F0, ray.dir, hit.normal);
            outRadiance = outRadiance + trace(Ray(hit.position + hit.normal * epsilon, reflectedDir), depth + 1) * F;
        }

        return outRadiance;
    }

    void Animate(float dt) { camera.Animate(dt); }
};

GPUProgram gpuProgram; // vertex and fragment shaders
Scene scene;

// vertex shader in GLSL
const char *vertexSource = R"(
	#version 330
    precision highp float;

	layout(location = 0) in vec2 cVertexPosition;	// Attrib Array 0
	out vec2 texcoord;

	void main() {
		texcoord = (cVertexPosition + vec2(1, 1))/2;							// -1,1 to 0,1
		gl_Position = vec4(cVertexPosition.x, cVertexPosition.y, 0, 1); 		// transform to clipping space
	}
)";

// fragment shader in GLSL
const char *fragmentSource = R"(
	#version 330
    precision highp float;

	uniform sampler2D textureUnit;
	in  vec2 texcoord;			// interpolated texture coordinates
	out vec4 fragmentColor;		// output that goes to the raster memory as told by glBindFragDataLocation

	void main() {
		fragmentColor = texture(textureUnit, texcoord);
	}
)";

class FullScreenTexturedQuad {
    unsigned int vao = 0, textureId = 0;    // vertex array object id and texture id
public:
    FullScreenTexturedQuad(int windowWidth, int windowHeight) {
        glGenVertexArrays(1, &vao);    // create 1 vertex array object
        glBindVertexArray(vao);        // make it active

        unsigned int vbo;        // vertex buffer objects
        glGenBuffers(1, &vbo);    // Generate 1 vertex buffer objects

        // vertex coordinates: vbo0 -> Attrib Array 0 -> vertexPosition of the vertex shader
        glBindBuffer(GL_ARRAY_BUFFER, vbo); // make it active, it is an array
        float vertexCoords[] = {-1, -1, 1, -1, 1, 1, -1, 1};    // two triangles forming a quad
        glBufferData(GL_ARRAY_BUFFER, sizeof(vertexCoords), vertexCoords,
                     GL_STATIC_DRAW);       // copy to that part of the memory which is not modified
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);     // stride and offset: it is tightly packed

        glGenTextures(1, &textureId);
        glBindTexture(GL_TEXTURE_2D, textureId);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    }

    void LoadTexture(std::vector<vec4> &image) {
        glBindTexture(GL_TEXTURE_2D, textureId);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, windowWidth, windowHeight, 0, GL_RGBA, GL_FLOAT, &image[0]);
    }

    void Draw() {
        glBindVertexArray(vao);    // make the vao and its vbos active playing the role of the data source
        int location = glGetUniformLocation(gpuProgram.getId(), "textureUnit");
        const unsigned int textureUnit = 0;
        if (location >= 0) {
            glUniform1i(location, textureUnit);
            glActiveTexture(GL_TEXTURE0 + textureUnit);
            glBindTexture(GL_TEXTURE_2D, textureId);
        }
        glDrawArrays(GL_TRIANGLE_FAN, 0, 4);    // draw two triangles forming a quad
    }
};

FullScreenTexturedQuad *fullScreenTexturedQuad;

// Initialization, create an OpenGL context
void onInitialization() {
    glViewport(0, 0, windowWidth, windowHeight);
    scene.build();

    // copy image to GPU as a texture
    fullScreenTexturedQuad = new FullScreenTexturedQuad(windowWidth, windowHeight);

    // create program for the GPU
    gpuProgram.create(vertexSource, fragmentSource, "fragmentColor");
}

// Window has become invalid: Redraw
void onDisplay() {
    std::vector<vec4> image(windowWidth * windowHeight);
    scene.render(image);
    fullScreenTexturedQuad->LoadTexture(image);
    fullScreenTexturedQuad->Draw();
    glutSwapBuffers();                                    // exchange the two buffers
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {

}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) {
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
    scene.Animate(0.1f);
    glutPostRedisplay();
}