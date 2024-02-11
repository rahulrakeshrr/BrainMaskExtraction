# BrainMaskExtraction
Removal of non-brain tissue from magnetic resonance imaging (MRI) data.

Problem: 
Extraction of the brain region from CTA images is often a manual and time-consuming
process. While there are existing opensource automatic brain extraction tools, such as the Brain
Extraction Tool (BET), these tools often face limitations when applied to CTA images. This is due to their
reliance on intensity-based methods, which may lead to the extraction of regions beyond the brain.
Moreover, a significant portion of CTA images also include the carotid arteries and the neck tissues,
further complicating the extraction process.

Image Format
The CTA images will be named 'CaseID.nii.gz' and their respective brain masks will be 'CaseID_ROI.nii.gz'
The brain masks will have either “0” or “1” as the intensity with “1” representing brain region.

Impact: 
CTA (Computed Tomography Angiography) is a medical imaging technique that combines Computed
tomography (CT) technology with the use of contrast agents to visualize blood vessels and flow of blood
throughout the body. MIP (Maximum Intensity Projection) images are created from CTA which will be
used to identify the location of vascular occlusion in the brain.

![image](https://github.com/rahulrakeshrr/BrainMaskExtraction/assets/83067337/085897bc-3d8c-4e39-9459-b9832ba92ac9)
![image](https://github.com/rahulrakeshrr/BrainMaskExtraction/assets/83067337/ed655118-f77c-4c2b-ba1a-62b63b2f395f)
![image](https://github.com/rahulrakeshrr/BrainMaskExtraction/assets/83067337/b94884b1-f4d7-4fbf-8b0e-3e66bab908a2)


