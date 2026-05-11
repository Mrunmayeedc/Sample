#include <iostream>
#include <omp.h>
#include <chrono>

using namespace std;
using namespace chrono;

void printArray(int arr[], int n) {
    for(int i=0;i<n;i++)
        cout << arr[i] << " ";
    cout << endl;
}

void copyArray(int src[], int dest[], int n) {
    for(int i=0;i<n;i++)
        dest[i] = src[i];
}

void sequentialBubbleSort(int arr[], int n) {
    for(int i=0;i<n-1;i++) {
        for(int j=0;j<n-i-1;j++) {
            if(arr[j] > arr[j+1])
                swap(arr[j], arr[j+1]);
        }
    }
}

void parallelBubbleSort(int arr[], int n) {
    for(int i=0;i<n;i++) {
        int start = i % 2;

        #pragma omp parallel for
        for(int j=start;j<n-1;j+=2) {
            if(arr[j] > arr[j+1])
                swap(arr[j], arr[j+1]);
        }
    }
}

void merge(int arr[], int left, int mid, int right) {
    int size = right-left+1;
    int* temp = new int[size];

    int i=left, j=mid+1, k=0;

    while(i<=mid && j<=right) {
        if(arr[i] <= arr[j])
            temp[k++] = arr[i++];
        else
            temp[k++] = arr[j++];
    }

    while(i<=mid)
        temp[k++] = arr[i++];

    while(j<=right)
        temp[k++] = arr[j++];

    for(int x=0;x<size;x++)
        arr[left+x] = temp[x];

    delete[] temp;
}

void sequentialMergeSort(int arr[], int left, int right) {
    if(left < right) {
        int mid = (left+right)/2;

        sequentialMergeSort(arr,left,mid);
        sequentialMergeSort(arr,mid+1,right);

        merge(arr,left,mid,right);
    }
}

void parallelMergeSort(int arr[], int left, int right) {
    if(left < right) {
        int mid = (left+right)/2;

        #pragma omp parallel sections
        {
            #pragma omp section
            parallelMergeSort(arr,left,mid);

            #pragma omp section
            parallelMergeSort(arr,mid+1,right);
        }

        merge(arr,left,mid,right);
    }
}

int main() {
    int n;

    cout << "Enter number of elements: ";
    cin >> n;

    int* original = new int[n];
    int* arr1 = new int[n];
    int* arr2 = new int[n];

    cout << "Enter array elements:\n";
    for(int i=0;i<n;i++)
        cin >> original[i];

    int choice;

    cout << "\n1. Bubble Sort";
    cout << "\n2. Merge Sort";
    cout << "\nEnter choice: ";
    cin >> choice;

    cout << "\nOriginal Array: ";
    printArray(original,n);

    switch(choice) {

        case 1: {
            copyArray(original,arr1,n);
            copyArray(original,arr2,n);

            auto t1 = high_resolution_clock::now();
            sequentialBubbleSort(arr1,n);
            auto t2 = high_resolution_clock::now();

            auto t3 = high_resolution_clock::now();
            parallelBubbleSort(arr2,n);
            auto t4 = high_resolution_clock::now();

            cout << "\nSequential Bubble: ";
            printArray(arr1,n);

            cout << "Parallel Bubble: ";
            printArray(arr2,n);

            cout << "\nSequential Time: "
                 << duration_cast<microseconds>(t2-t1).count()
                 << " microseconds";

            cout << "\nParallel Time: "
                 << duration_cast<microseconds>(t4-t3).count()
                 << " microseconds";
            break;
        }

        case 2: {
            copyArray(original,arr1,n);
            copyArray(original,arr2,n);

            auto t5 = high_resolution_clock::now();
            sequentialMergeSort(arr1,0,n-1);
            auto t6 = high_resolution_clock::now();

            auto t7 = high_resolution_clock::now();
            parallelMergeSort(arr2,0,n-1);
            auto t8 = high_resolution_clock::now();

            cout << "\nSequential Merge: ";
            printArray(arr1,n);

            cout << "Parallel Merge: ";
            printArray(arr2,n);

            cout << "\nSequential Time: "
                 << duration_cast<microseconds>(t6-t5).count()
                 << " microseconds";

            cout << "\nParallel Time: "
                 << duration_cast<microseconds>(t8-t7).count()
                 << " microseconds";
            break;
        }

        default:
            cout << "\nInvalid choice!";
    }

    delete[] original;
    delete[] arr1;
    delete[] arr2;

    return 0;
}
