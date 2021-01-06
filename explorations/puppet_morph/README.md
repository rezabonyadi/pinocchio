# face_morph
 
This is a simple, opencv based, face morphing. 

Problem: You have an image s and an image d, both ocntain an arbitrary face in them. You want the style of the face d is transfered to the face s, e.g., if d is laghing you want s to lagh.

Given two portraits, s and d, we want d to have the same style as of s. For example, if s has a face with open mouth, we want the face in d to have an open mouth too, eventhough the mouth in d is close in the original image. This can be done by deep learning tools and style transfers (see face swap in my other library in GitHub), but here it is done by image morphing.

1) Find the images landmarks (deep learning is used here) 
2) Find Delauney triangulation of the faces built on top of the landmarks for d. Essentially, built the triangulation for d, use the same order of landmarks to build the triangles on d for s as well. This makes sure that the triangles point to the same areas as of in d.
3) For each pair of triangles (in d and s, note they are corresponding as they have been built on the same landmark orders), find a transformation from trangle in d to the same triangle in s.

To install use the environment.yml 
