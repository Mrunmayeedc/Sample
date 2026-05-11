#include <omp.h>
#include <iostream>
#include <chrono>

using namespace std;
using namespace std::chrono;

// Display Array
void displayArray(int nums[], int length)
{
    cout << "Nums: [";
    for (int i = 0; i < length; i++) {
        cout << nums[i];
        if (i != length - 1)
            cout << ", ";
    }
    cout << "]" << endl;
}

// Parallel Min
void minOperation(int nums[], int length)
{
    int minValue = nums[0];

    #pragma omp parallel for reduction(min:minValue)
    for (int i = 0; i < length; i++) {
        if (nums[i] < minValue)
            minValue = nums[i];
    }

    cout << "Min value: " << minValue << endl;
}

// Parallel Max
void maxOperation(int nums[], int length)
{
    int maxValue = nums[0];

    #pragma omp parallel for reduction(max:maxValue)
    for (int i = 0; i < length; i++) {
        if (nums[i] > maxValue)
            maxValue = nums[i];
    }

    cout << "Max value: " << maxValue << endl;
}

// Parallel Sum
void sumOperation(int nums[], int length)
{
    int sum = 0;

    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < length; i++) {
        sum += nums[i];
    }

    cout << "Sum: " << sum << endl;
}

// Parallel Average
void avgOperation(int nums[], int length)
{
    float sum = 0;

    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < length; i++) {
        sum += nums[i];
    }

    cout << "Average: " << (sum / length) << endl;
}

// Main
int main()
{
    int n;

    cout << "Enter number of elements: ";
    cin >> n;

    int* nums = new int[n];

    cout << "Enter array elements:\n";
    for (int i = 0; i < n; i++) {
        cin >> nums[i];
    }

    auto start = high_resolution_clock::now();

    displayArray(nums, n);
    minOperation(nums, n);
    maxOperation(nums, n);
    sumOperation(nums, n);
    avgOperation(nums, n);

    auto stop = high_resolution_clock::now();

    auto duration = duration_cast<microseconds>(stop - start);

    cout << "\nExecution time: "
         << duration.count()
         << " microseconds" << endl;

    delete[] nums;

    return 0;
}
