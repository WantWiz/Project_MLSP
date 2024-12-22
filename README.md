Ce repo se base sur le travail effectué par Pablo Alonso-Jiménez pour le projet PECMAE :https://github.com/palonso/pecmae/ 

Les modifications que nous avons apporté sont les suivantes :

- Ajout du dossier /features qui contient les features extraites des musiques du dataset Gtzan par l'encoder EncodecMAE. Chacune des musiques a subi une séparation des sources (bass, drums, vocals, other) et nous avons extrait les features de chacune de ces composantes.
- Dans le dossier /src : 
    - le dossier /output_linear_fusion_model contient les outputs et les poids de notre modèle de classification par fusion linéaire
    - le dossier /output_prototype_model contient les outputs et les poids de notre modèle principale de classification par comparaison avec des prototypes
    - ajout du fichier train_multi_prototypes.py qui permet d'entraîner notre modèle principale de classification par comparaison avec des prototypes
    - ajout du fichier train_multi_linear.py qui permet d'entraîner notre modèle principale de classification par par fusion linéaire
    - ajout du fichier test.py qui permet de tester le modèle de classification par fusion linéaire
    - ajout du fichier classify_song.py qui permet de classifier une nouvelle musique au format mp3
- Le dossier /decoded_proto qui contient les différents prototypes appris pour la classification au format wav