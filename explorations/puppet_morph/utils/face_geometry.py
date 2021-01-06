import numpy as np
import cv2
import face_alignment
import face_recognition
import dlib

class FaceGeometry:
    '''
    This class generates geometry (landmark points, Delauney triangles, and bounding box) for a face in a given image.
    '''
    def __init__(self, image, landmarks_model='recog', perc_out_face=0.1):
        '''
        # image: Is the input image, assumes to contain at least a face.
        # landmarks_model: The model to be used to generate landmarks. You may use face_recognition (recog), 
                            face_alignment (align), dlib (dlib), or a file address (.txt).
        # perc_out_face: Stretches the face rectangle to include some of the outer patterns from the face.
        
        '''        
        self.image = image
        self.landmarks_model = landmarks_model
        self.landmarks_from_file = True if  '.' in landmarks_model else False
        self.perc_out_face = perc_out_face
        
        self.face_triangles_index = None # Triangles index corresponding with the face_landmarks_points
        self.face_landmarks = None # face landmarks as list
        self.face_landmarks_points = None # Face landmarks as points
        self.face_rect = None # Face bounding box, with stretch set by perc_out_face
        self.face_landmarks_points_with_corners = None # Face landmarks with corner points
        
    def build_geometry(self):
        # Builds the geometry
        
        self.face_landmarks, self.face_landmarks_points = self.calculate_landmarks()
        self.face_rect = self.find_face_rect(self.face_landmarks)  
        self.face_landmarks_points_with_corners = self.add_corners(self.face_landmarks_points, self.face_rect)
        self.face_triangles_index = self.calculate_delaunay_triangles(self.face_rect, self.face_landmarks_points_with_corners)
        
        return self
    
    def calculate_landmarks(self):
        # Creats the landmarks, given the image in self.image, landmark calculation model set in self.landmarks_model
        img = self.image 
        model_source = self.landmarks_model 
        from_file=self.landmarks_from_file
        
        if from_file: # If from file, then just read the file
            landmarks_file = model_source
            f = open(landmarks_file, "r")
            Lines = f.readlines() 

            count = 0
            face_landmarks = []
            for line in Lines: 
                x, y, _ = line.split(' ')
                face_landmarks.append([int(float(x)), int(float(y))])
            f.close()
        else:
            if model_source is 'align':
                fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)
                faces_landmarks = fa.get_landmarks(img)
                
                assert faces_landmarks != None, 'No face was found in the image!'
                assert len(faces_landmarks)!=0, 'No face was found in the image!'
                print('More than one face was found. The first one is considered only!') if len(faces_landmarks)>1 else None
                
                face_landmarks = faces_landmarks[0]  # Only first face

            if model_source is 'recog':
                faces_landmarks_list = face_recognition.face_landmarks(img)
                face_landmarks = []
                
                assert len(faces_landmarks_list)!=0, 'No face was found in the image!'
                print('More than one face was found. The first one is considered only!') if len(faces_landmarks_list)>1 else None
                
                face_landmarks_list = faces_landmarks_list[0]
                
                for face_component in face_landmarks_list.values():
                    face_landmarks.extend(list(face_component))    
                face_landmarks = face_landmarks[:-4]
                face_landmarks = np.array(face_landmarks, np.int32)

            if model_source is 'dlib':
                detector = dlib.get_frontal_face_detector()
                predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = detector(img_gray)
                
                assert len(faces)!=0, 'No face was found in the image!'
                
                face = faces[0] # Only takes into account the first face
                print('More than one face ws found. The first one is considered only!') if len(faces)>1 else None
                
                landmarks = predictor(img_gray, face)

                face_landmarks = []
                for n in range(0, 68):
                    x = landmarks.part(n).x
                    y = landmarks.part(n).y
                    face_landmarks.append([x, y])
                face_landmarks = np.array(face_landmarks, np.int32)
                
        face_landmarks = np.array(face_landmarks, np.int32)    
        landmark_points = [(int(lnd[0]), int(lnd[1])) for lnd in face_landmarks] 
        
        # face_landmarks as list, landmark_points as points
        return face_landmarks, landmark_points


    def find_face_rect(self, face_landmarks):
        '''
        Generates the face bounding box, made upon the convex hul of the face landmarks. It also stretches the bounding box
        to include a bit over the face.
        
        # face_landmarks: Face landmarks        
        '''        
        
        perc_out_face = self.perc_out_face
        image = self.image
        
        convexhull = cv2.convexHull(face_landmarks)
        rect = cv2.boundingRect(convexhull)    
        # Stretch the bounding box a bit to include more of the face. This is not needed if the face swap is the aim.
        # This is useful though if the style is going to transfer.
        if perc_out_face>0.0:    
            w = rect[2]
            h = rect[3]
            t = np.max([rect[1]-np.int32(h*perc_out_face), 0])
            r = np.min([rect[0]+w+np.int32(w*perc_out_face), image.shape[1]-1])
            b = np.min([rect[1]+h+np.int32(h*perc_out_face), image.shape[0]-1])
            l = np.max([rect[0]-np.int32(w*perc_out_face), 0])

            w = r-l
            h = b-t

            rect = (int(l), int(t), int(w), int(h))
        
        # The bounding box
        return rect


    def add_corners(self, landmark_points, face_rect):
        '''
        Add corners of the face rect. This is in particular usefulr for style transfer.
        '''
        l = face_rect[0]+1
        t = face_rect[1]+1
        r = face_rect[0]+face_rect[2]-1
        b = face_rect[1]+face_rect[3]-1
        lt = (l, t)
        rt = (r, t)
        lb = (l, b)
        rb = (r, b)
        mt = (l+int((r-l)/2.0), t)
        ml = (l, t+int((b-t)/2.0))
        mb = (l+int((r-l)/2.0), b)
        mr = (r, t+int((b-t)/2.0))

        landmark_points.extend([lt, rt, lb, rb, mt, ml, mb, mr]) 

        return landmark_points


    def calculate_delaunay_triangles(self, face_rect, landmark_points):   
        '''
        Find Delaunay triangles, indexed on the landmarks points.
        '''
        image = self.image
        
        subdiv = cv2.Subdiv2D(face_rect)

        # Now build the Delauney triangles
        subdiv.insert(landmark_points)
        triangles = subdiv.getTriangleList()
        triangles = np.array(triangles, dtype=np.int32)

        indexes_triangles = []
        for t in triangles:
            pt1 = (t[0], t[1])
            pt2 = (t[2], t[3])
            pt3 = (t[4], t[5])
            index_pt1 = landmark_points.index(pt1)
            index_pt2 = landmark_points.index(pt2)
            index_pt3 = landmark_points.index(pt3)
            if index_pt1 is not None and index_pt2 is not None and index_pt3 is not None:
                triangle = [index_pt1, index_pt2, index_pt3]
                indexes_triangles.append(triangle)

        return indexes_triangles
    
    def visualize(self, overlay_on_image=True):
        '''
        Visualize the landmarks, triangles, and bounding box, overlayed on the image (or not!)
        '''
        img = self.image 
        landmark_points = self.face_landmarks_points_with_corners 
        triangles = self.face_triangles_index
        
        if overlay_on_image:
            img_copy = img.copy()
        else:
            img_copy = np.zeros((img.shape))

        for p in landmark_points:
            cv2.circle(img_copy, p, 2, (0, 255, 0), -1)

        if triangles is not None:
            delaunay_color = (0, 0, 0)

            for t in triangles:
                pt1 = landmark_points[t[0]]
                pt2 = landmark_points[t[1]]
                pt3 = landmark_points[t[2]]
                cv2.line(img_copy, pt1, pt2, delaunay_color, 1)
                cv2.line(img_copy, pt2, pt3, delaunay_color, 1)
                cv2.line(img_copy, pt3, pt1, delaunay_color, 1)

        cv2.imshow('Press any key to exit', img_copy)
        cv2.waitKey(0)
        cv2.destroyAllWindows()