#include <iostream>
#include <vector>
#include <queue>
#include <unordered_map>
#include <functional>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <random>

// Function to partition the array for quickselect
int partition(std::vector<float>& nums, int left, int right, int pivotIndex) {
    float pivotValue = nums[pivotIndex];
    std::swap(nums[pivotIndex], nums[right]);
    int storeIndex = left;

    for (int i = left; i < right; i++) {
        if (nums[i] > pivotValue) { // Changed to > for finding the largest elements
            std::swap(nums[storeIndex], nums[i]);
            storeIndex++;
        }
    }
    std::swap(nums[right], nums[storeIndex]);
    return storeIndex;
}

// Optimized Quickselect to approximate top-K elements
void quickselect(std::vector<float>& nums, int left, int right, int k) {
    while (left < right) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(left, right);

        int pivotIndex = dis(gen);
        pivotIndex = partition(nums, left, right, pivotIndex);

        if (k == pivotIndex) {
            return;
        } else if (k < pivotIndex) {
            right = pivotIndex - 1;
        } else {
            left = pivotIndex + 1;
        }
    }
}

// Define a hash function for floating-point numbers
struct FloatHash {
    size_t operator()(float value) const {
        // Use reinterpret_cast to convert the float to an int representation for hashing
        int intValue = *reinterpret_cast<int*>(&value);
        return std::hash<int>{}(intValue);
    }
};

class DynamicBuckets {
public:
    DynamicBuckets(size_t numBuckets, size_t threshold)
        : numBuckets(numBuckets), threshold(threshold), buckets(numBuckets) {}

    // Function to quickly process all elements
    void processAllElements(std::vector<float>& elements) {
        auto start = std::chrono::high_resolution_clock::now();

        // Use quickselect to find the top-K largest elements
        quickselect(elements, 0, elements.size() - 1, threshold);

        // Clear the buckets and redistribute elements
        for (auto& bucket : buckets) {
            while (!bucket.empty()) {
                bucket.pop();
            }
        }

        // Redistribute top K elements into the first bucket
        for (size_t i = 0; i < threshold; ++i) {
            buckets[0].push(i);
        }

        // Distribute the remaining elements into the other buckets
        for (size_t i = threshold; i < elements.size(); ++i) {
            size_t bucketIndex = (i - threshold) % (numBuckets - 1) + 1;
            buckets[bucketIndex].push(i);
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;
        std::cout << "Time taken to process all elements: " << duration.count() << " seconds" << std::endl;
    }

    // Function to incrementally process each element
    void addElement(size_t index) {
        size_t bucketIndex = hashFunction(index) % numBuckets;
        buckets[bucketIndex].push(index);
        if (buckets[bucketIndex].size() > threshold) {
            splitBucket(bucketIndex);
        }
    }

    void printBuckets(const std::vector<float>& elements) {
        for (size_t i = 0; i < buckets.size(); ++i) {
            std::cout << "Bucket " << i << " (" << buckets[i].size() << " elements): ";
            if (!buckets[i].empty()) {
                std::queue<size_t> tempQueue = buckets[i];
                std::cout << elements[tempQueue.front()] << " ";
                tempQueue.pop();
                if (!tempQueue.empty()) {
                    std::cout << elements[tempQueue.front()] << " ";
                    tempQueue.pop();
                }
                if (!tempQueue.empty()) {
                    std::cout << "... ";
                    while (tempQueue.size() > 1) {
                        tempQueue.pop();
                    }
                    std::cout << elements[tempQueue.front()];
                }
            }
            std::cout << std::endl;
        }
    }

    // Remove and redistribute elements from the first bucket
    void removeAndRedistributeFirstBucket() {
        std::queue<size_t> temp = std::move(buckets[0]);
        while (!buckets[0].empty()) {
            buckets[0].pop();
        }
        while (!temp.empty()) {
            addElement(temp.front());
            temp.pop();
        }
    }

private:
    size_t numBuckets;
    size_t threshold;
    std::vector<std::queue<size_t>> buckets;
    FloatHash hashFunction;

    void splitBucket(size_t bucketIndex) {
        std::queue<size_t>& bucket = buckets[bucketIndex];
        std::queue<size_t> newBucket;
        size_t newBucketIndex = buckets.size();
        buckets.push_back(newBucket);

        while (!bucket.empty()) {
            size_t index = bucket.front();
            bucket.pop();
            if (hashFunction(index) % numBuckets != bucketIndex) {
                buckets[newBucketIndex].push(index);
            } else {
                bucket.push(index);
            }
        }

        ++numBuckets; // Increase the number of buckets after splitting
    }
};

float generateRandomFloat(float min, float max) {
    float random = ((float) rand()) / (float) RAND_MAX;
    float range = max - min;  
    return (random * range) + min;
}

int main() {
    size_t vert = 160000;
    size_t numBuckets = 100;
    size_t bucketThreshold = 10000; // Adjust threshold according to your needs
    int w = 0; // The constant w

    // Initialize random seed
    srand(static_cast<unsigned int>(time(0)));

    // Simulate R[w][i] as a 2D vector where w is constant
    std::vector<std::vector<float>> R(1, std::vector<float>(vert));

    // Generate random float numbers for R[w][i]
    for (size_t i = 0; i < vert; ++i) {
        R[w][i] = generateRandomFloat(0.0, 100.0);
    }

    // Create an instance of DynamicBuckets
    DynamicBuckets dynamicBuckets(numBuckets, bucketThreshold);

    // Process all elements from R[w]
    dynamicBuckets.processAllElements(R[w]);

    // Print the buckets after processing all elements
    dynamicBuckets.printBuckets(R[w]);

    // Remove and redistribute the first bucket
    dynamicBuckets.removeAndRedistributeFirstBucket();
    std::cout << "\nFirst bucket removed and redistributed.\n";

    // Add 40,000 new elements incrementally with timing
    std::cout << "\nAdding 40,000 new elements incrementally:\n";
    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = vert; i < vert + 40000; ++i) {
        float newValue = generateRandomFloat(0.0, 100.0);
        R[w].push_back(newValue);
        dynamicBuckets.addElement(i);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "Time taken to add 40,000 new elements incrementally: " << duration.count() << " seconds" << std::endl;

    // Print the buckets after adding new elements incrementally
    dynamicBuckets.printBuckets(R[w]);

    return 0;
}
