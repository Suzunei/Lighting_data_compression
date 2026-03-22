5个.py文件分别为：  
MBD.py ->MBD+Gauss  
MBD_Control.py -> MBD  
gaussian_control.py -> Gauss  
gaussian.py -> Gauss + MLP  
MBD_Demo.py -> demo for testing  

Output中分别为：  
3Dgaussian_only.png ->3D Gauss  
MBD_control_3D.png  ->MBD  
MBD+3Dgaussian.png  ->MBD +3D Gauss  
3Dgaussian+MLP_2000epochs.png &   
  3Dgaussian+MLP_1500epochs.png -> UnFreeze the Gaussian during the quantization stage.  
3Dgaussian+MLP_1400+100epochs.png &  
3Dgaussian+MLP_1100+400epochs.png &  
3Dgaussian+MLP_1200+300epochs.png -> Freeze the Gaussian during the quantization stage.  
