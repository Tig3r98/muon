//****************************************************************************
/*    SCALER
gcc SCA.c -Wall -o SCA -lxx_usb -lusb

Descrizione:
Il programma legge e stampa (a terminale e su file) i 12 registri dello
scaler posto nella station numerata con SCA_SLOT.
La misura dura TIME e stampa ogni TIME_STAMP.

*/
//****************************************************************************

// N è il numero della stazione nel crate (0-24)
// A è il subadress (usato per moduli a più sezioni)
// F è il numero della funzione

#define SCA_SLOT 21
#define TIME 60         //tempo di misura in secondi
#define TIME_STAMP 10	//ogni quanto tempo il programma stampa

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/mman.h>
#include <errno.h>
#include <ctype.h>

#include <libxxusb.h>

//find camac
usb_dev_handle* find_devices(){
    //maximum number of crates: 128
    xxusb_device_type devices[128];
    
    short count = xxusb_devices_find(devices);
    if (count < 0)
      printf("Couldn't get list of XXUSB (crate) devices.");
    
    for (int i=0; i<count; ++i) {
        auto* dev = devices[i].usbdev;
        //open device
        usb_dev_handle* handle = xxusb_device_open(dev);
        //verify correct opening
        if(!handle)
            return 0;
        return handle;
    }
    return 0;
}

usb_dev_handle* handle;
char *cszPrgName;

int main(int argc, char **argv){
  //variabili necessaria
  char *fname = DEVICE_NAME;
  long int i , j, c;
  int error;
  long k;
  int q, x;

  // apertura del file di output dei dati
  FILE *dati = fopen("./dati/datiscaler.txt","w");
  if (dati == NULL){
    perror("Errore apertura file: manca la cartella?");
    exit(1);
  }

  //apertura crate usb
  handle = find_devices();
  if(handle == 0){
    fprintf(stderr, "%s: %s: %s\n", cszPrgName, fname, strerror(error));
    exit(1);
  } else{
    printf("%s: open done.\n", cszPrgName);
  }
  
for (c=0; c<5; c++){		//Ripete la presa dati per varie volte

  // Invia un comando allo scaler nel sottocanale A
  // input: handle, N, A, F

  for (i = 0; i < 12; i++) CAMAC_read(handle,SCA_SLOT,i,2,&k,&q,&x);

  // Lettura dallo scaler

  // i va da zero a un numero alto che non dovrebbe essere raggiunto

  for (i = 1; i < (2*TIME/TIME_STAMP); i++) {
    sleep(TIME_STAMP);
    printf("%ld Scaler counts: ",TIME-i*TIME_STAMP);   // print time left
    for (j = 0; j < 12; j++) {                 // per i 12 registri dello scaler
      CAMAC_read(handle,SCA_SLOT,j,0,&k,&q,&x); // lettura registri e stampa
      printf("%ld ",k);
      fprintf(dati, "%ld ", k);
    }
    if (i*TIME_STAMP >= TIME) break;  //se si è superato TIME si esce
    printf("\n");
    fprintf(dati, "\n");

  }

  //stampa dei risultati

  fprintf(dati," \nTempo di aquisizione: %ld secondi",i*TIME_STAMP);
  printf("Tempo di aquisizione: %ld secondi",i*TIME_STAMP);
  printf("\n");
}

  //chiude il canale di comunicazione e il file
  xxusb_device_close(handle);
  printf("%s: close done.\n", cszPrgName);
  fclose(dati);

  return 0;
}
