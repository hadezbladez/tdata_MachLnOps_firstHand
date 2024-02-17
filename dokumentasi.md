# Submission 1: Nama Proyek Anda
Nama: Handerson Loriano

Username dicoding: hadezbladez

| | Deskripsi |
| ----------- | ----------- |
| Dataset | [*Skin Cancer Dataset*](https://www.kaggle.com/datasets/farjanakabirsamanta/skin-cancer-dataset) |
| Masalah | alat dermatoscope adalah alat yang mempunyai biaya yang cukup besar sehingga ini menjadi pertimbangan untuk para ahli menggunakan alatnya.|
||Tidak mudah bagi manusia yang ahli mendiagnosa kulit dari kasat mata karena, alat *dermatoscopy* membutuhkan penilaian dan pertimbangan dari mata manusia sehingga ini menjadi hal yang perlu diselesaikan. Untungnya beberapa gambar dari alat *dermatoscopy* sangat sesuai untuk *machine learning*
||Sampel data yang tidak terlalu banyak dapat menyebabkan hasil analisa *AI* yang tidak sesuai karenanya, dimunculkan data-data pasien dari populasi yang berbeda-beda, maka dibentuklah dataset yang terbaru ini karena, dataset sebelumnya tidak mempunyai keunikan yang berbeda-beda dan sampelnya sangat sedikit
|Isi Dataset|lesion_id = identifikasi unik
||image_id = identifikasi gambar yang berhubungan
||dx = Nama penyakit kanker kulit. (Sebagai Label / Y)
||dx_type = Dasar Ukuran Penyakit kanker kulit
||age = umur (Feature / X)
||sex = jenis kelamin (Feature / X)
||localization = daerah kulit (Feature / X)
| Solusi machine learning | *Deep Learning - Tensorflow* menggunakan *hyperparameter* *learning-rate* berupaya memaksimalkan akurasi dari model|
|| Dataset Ini memiliki versi yang sebelumnya maka dari itu sangat cocok digunakan karena, dengan membanding proses sebelumnya kita bisa mempunyai ukuran perbandingan yang tepat
| Metode pengolahan | Rangkuman |
| - Tahap *Data Ingestion*| *example_gen_pb2* memakai perbandingan *Train:Eval* = 8:2 untuk *CsvExampleGen*
| - Tahap *Data Validation*| Menggunakan *SchemaGen* dan *ExampleValidator*
| - Tahap *Data Preprocessing*|Pada Tahap *Transform* menggunakan *sparse_tensor_to_dense_with_shape*, *compute_and_apply_vocabulary*, *one-hot* dan *Module_File*
| - Tahap *Modelling*| Menggunakan *Tuner* untuk memaksimalkan daya *learning_rate*
||Untuk *Trainer* Memakai cara *Module_File* 
|- Tahap *Evaluation*| Model Akan dievaluasi menggunakan *Binary Accuracy* dan turunan perhitungannya
| Arsitektur model | 
|- [*Input Layer*](https://www.tensorflow.org/api_docs/python/tf/keras/layers/InputLayer)| Ini *layer* untuk menerima data yang kita dapatkan dari *transform* 3 untuk *feature* yang akan diterima model
|- [*Layer Category Encoding*](https://www.tensorflow.org/api_docs/python/tf/keras/layers/CategoryEncoding)|Setelah mendapat input *Layer* ini akan menerima sesuai kategori *encode* untuk data kategorikal
|- [*Layer Nest Flatten*](https://www.tensorflow.org/api_docs/python/tf/nest/flatten)| *Layer* yang terpisah akan dikompres menjadi layer yang kecil
|- [concat layer](https://www.tensorflow.org/tfx/tutorials/transform/census)| *Layer* tersebut akan dijadikan satu
|- [*Layer Dense*](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense)| *Layer* Normal yang menggunakan Aktivasi relu dan sigmoid untuk output|
| Metrik evaluasi | Menggunakan *Confusion Matrix* nilai data :
||*ExampleCount*
||*AUC*
||*FalsePositives*
||*TruePositives*
||*FalseNegatives*
||*TrueNegatives*
||*BinaryAccuracy*|
| Performa model | 80% |