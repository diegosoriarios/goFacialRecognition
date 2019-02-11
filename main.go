package main

import (
	"fmt"
	"log"
	"path/filepath"
	"github.com/Kagami/go-face"
)

const dataDir = "images"

func main(){
	fmt.Println("Facial Recognition System v0.01")

	rec, err := face.NewRecognizer(dataDir)

	if err != nil {
		fmt.Println("Error Creating Recognizer")
		fmt.Println(err)
	}

	defer rec.Close()

	avengersImage := filepath.Join(dataDir, "avengers-02.jpeg")

	faces, err := rec.RecognizeFile(avengersImage)

	if err != nil {
		log.Fatalf("Can't recognize file")
	}

	fmt.Println("Number of faces in Image: ", len(faces))

	var samples []face.Descriptor
	var avengers []int32
	for i, f := range faces {
		samples = append(samples, f.Descriptor)
		avengers = append(avengers, int32(i))
	}

	labels := []string {
		"Dr Strange",
		"Tony Stark",
		"Bruce Banner",
		"Wong",
	}

	rec.SetSamples(samples, avengers)

	testTonyStark := filepath.Join(dataDir, "tony-stark.jpg")
	tonyStark, err := rec.RecognizeSingleFile(testTonyStark)
	if err != nil {
		log.Fatalf("Faced error with file")
	}

	avengerId := rec.Classify(tonyStark.Descriptor)
	if avengerId < 0 {
		log.Fatalf("Can't Classify based of existing database")
	}

	fmt.Println(avengerId)
	fmt.Println(labels[avengerId])
}