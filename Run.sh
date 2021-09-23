while true
do
   echo "Running blender"
   blender Generator.blend -b -P renderScenes.py 
   echo "Crushed"
done

