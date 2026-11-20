/* Minimal MPSGraph stubs for the Linux ObjC syntax gate (make objc-syntax).
 * Selectors mirror Apple's documented MPSGraph API (verified against the
 * developer.apple.com class reference where possible). Declarations only;
 * never compiled into any real build. */
#ifndef H3I_STUB_MPSGRAPH_H
#define H3I_STUB_MPSGRAPH_H

#import <Metal/Metal.h>

typedef NSUInteger MPSDataType;
enum {
    MPSDataTypeFloat32 = 0x10000000u | 32u,
    MPSDataTypeBFloat16 = 0x10000000u | 16u,
};

@interface MPSGraphTensor : NSObject
@end

@interface MPSGraphTensorData : NSObject
- (instancetype)initWithMTLBuffer:(id<MTLBuffer>)buf
                            shape:(NSArray<NSNumber *> *)shape
                         dataType:(MPSDataType)dt;
@end

@interface MPSGraph : NSObject
- (MPSGraphTensor *)placeholderWithShape:(NSArray<NSNumber *> *)shape
                                dataType:(MPSDataType)dt
                                    name:(NSString *)name;
- (MPSGraphTensor *)constantWithData:(NSData *)data
                               shape:(NSArray<NSNumber *> *)shape
                            dataType:(MPSDataType)dt;
- (MPSGraphTensor *)transposeTensor:(MPSGraphTensor *)t
                          dimension:(NSInteger)d1
                      withDimension:(NSInteger)d2
                               name:(NSString *)name;
- (MPSGraphTensor *)matrixMultiplicationWithPrimaryTensor:(MPSGraphTensor *)a
                                          secondaryTensor:(MPSGraphTensor *)b
                                                     name:(NSString *)name;
- (MPSGraphTensor *)additionWithPrimaryTensor:(MPSGraphTensor *)a
                              secondaryTensor:(MPSGraphTensor *)b
                                         name:(NSString *)name;
- (MPSGraphTensor *)multiplicationWithPrimaryTensor:(MPSGraphTensor *)a
                                    secondaryTensor:(MPSGraphTensor *)b
                                               name:(NSString *)name;
- (MPSGraphTensor *)sigmoidWithTensor:(MPSGraphTensor *)t name:(NSString *)name;
- (MPSGraphTensor *)softMaxWithTensor:(MPSGraphTensor *)t
                                 axis:(NSInteger)axis
                                 name:(NSString *)name;
- (MPSGraphTensor *)sliceTensor:(MPSGraphTensor *)t
                         starts:(NSArray<NSNumber *> *)starts
                           ends:(NSArray<NSNumber *> *)ends
                        strides:(NSArray<NSNumber *> *)strides
                           name:(NSString *)name;
- (NSDictionary<MPSGraphTensor *, MPSGraphTensorData *> *)
    encodeToCommandBuffer:(id<MTLCommandBuffer>)cb
                    feeds:(NSDictionary<MPSGraphTensor *, MPSGraphTensorData *> *)src
         targetOperations:(NSArray *)ops
        resultsDictionary:(NSDictionary<MPSGraphTensor *, MPSGraphTensorData *> *)dst
    executionDescriptor:(id)desc;
@end

#endif /* H3I_STUB_MPSGRAPH_H */
