#include <unistd.h>
#include "unet.h"
#include <time.h>
#include <iostream>
void dealloc(void* data, size_t len, void* arg)
{
    free(data);
}

void free_buffer(void* data, size_t len)
{
    free(data);
}

TF_Buffer* read_tf_buffer(const char* file) {                                                  
    FILE *f = fopen(file, "rb");
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);                                                                  
    fseek(f, 0, SEEK_SET);  //same as rewind(f);                                            
    void* data = malloc(fsize);                                                             
    fread(data, fsize, 1, f);
    fclose(f);

    TF_Buffer* buf = TF_NewBuffer();                                                        
    buf->data = data;
    buf->length = fsize;                                                                    
    buf->data_deallocator = free_buffer;                                                    
    return buf;
} 

UNet::UNet(int64_t __n) : n(__n), w(320), h(240), c(3), dims{__n, h, w, c}
{
    printf("Found TensorFlow version %s\n", TF_Version());

    if (access("unet.pb", F_OK ) == -1)
    {
        fprintf(stderr, "unet.pb does not exist. Please freeze a graph for inference.\n");
    }
    
    s = n * h * w * c;

    graph = TF_NewGraph();
    status = TF_NewStatus();
    
    TF_Buffer* graph_def = read_tf_buffer("unet.pb");                      
    
    TF_SessionOptions* sopts = TF_NewSessionOptions();

    // Import graph_def into graph                                                          
    TF_ImportGraphDefOptions* opts = TF_NewImportGraphDefOptions();                         
    TF_GraphImportGraphDef(graph, graph_def, opts, status);
    
    const char* msg = TF_Message(status);
    int code = TF_GetCode(status);
    printf("\nLoaded graph. Status: %s\n\n", code ? msg : "SUCCESS");

    sess = TF_NewSession(graph, sopts, status);
    TF_DeleteSessionOptions(sopts);
    TF_DeleteImportGraphDefOptions(opts);
    TF_DeleteBuffer(graph_def);

    msg = TF_Message(status);
    code = TF_GetCode(status);
    printf("\nStarted session. Status: %s\n\n", code ? msg : "SUCCESS");
    
    in_op = {TF_GraphOperationByName(graph, 
            "UNet/images"), 0};
    if (!in_op.oper) {
        fprintf(stderr, "Can't init in_op.");
    } else {
        printf("Successfully initialized in_op\n");
    }

    out_op[0] = {TF_GraphOperationByName(graph, "UNet/mask"), 0};
    out_op[1] = {TF_GraphOperationByName(graph, "UNet/bbox"), 0};
    
    if (!out_op[0].oper || !out_op[1].oper) {
        fprintf(stderr, "Can't init out_op.");
    } else {
        printf("Successfully initialized out_op\n");
    }
    
    float* dummy_data;
   
    dummy_data = (float*)malloc(s*sizeof(float));
    memset(dummy_data, 0.0f, s*sizeof(float));

    input = TF_NewTensor(
        TF_FLOAT, dims,
        4, dummy_data, s*sizeof(float),
        &dealloc, NULL);

    // Warmup run to let TF optimize
    TF_SessionRun(sess,
                  NULL, // Run options.
                  &in_op, &input, 1, // Input tensors, input tensor values, number of inputs.
                  out_op, output, 2, // Output tensors, output tensor values, number of outputs.
                  NULL, 0, // Target operations, number of targets.
                  NULL, // Run metadata.
                  status // Output status.
    );
    
    if (input)
    {
        TF_DeleteTensor(input);
        input = NULL;
    }
    if (output[0])
    {
        TF_DeleteTensor(output[0]);
        output[0] = NULL;
    }
    if (output[1])
    {
        TF_DeleteTensor(output[1]);
        output[1] = NULL;
    }

    printf("Successfully ran warm-up\n");
}


UNet::~UNet()
{

    if (sess)
    {
        TF_CloseSession(sess, status);
        TF_DeleteSession(sess, status);
    }
    if (input)
        TF_DeleteTensor(input);
    if (output[0])
        TF_DeleteTensor(output[0]);
    if (output[1])
        TF_DeleteTensor(output[1]);
    if (status)
        TF_DeleteStatus(status);
    printf("Successfully closed session and session data\n");

}

std::vector<cv::Rect> UNet::run(const std::vector<cv::Mat>& _im, std::vector<cv::Mat>& out)
{   
    if (input)
    {
        TF_DeleteTensor(input);
        input = NULL;
    }
    if (output[0])
    {
        TF_DeleteTensor(output[0]);
        output[0] = NULL;
    }
    if (output[1])
    {
        TF_DeleteTensor(output[1]);
        output[1] = NULL;
    }

    cv::Size sz(w, h);
    cv::Mat im;
    float* data = (float*)malloc(s*sizeof(float));
    for (int i = 0; i < n; i++)
    {
        cv::resize(_im[i], im, sz);
        im.convertTo(im, CV_32F);
        im /= 255.0;
        // Avoid double free with opencv and tf.
        memcpy(data + i*h*w*c, im.data, h*w*c*sizeof(float));
    }
    /*
    for (int k = 0; k < c; k++)
    {
        for (int i = 0; i < h; i++)
        {
            for (int j = 0; j < w; j++)
                printf("%.3f ", data[k*w*h + i*w + j]);
            printf("\n");
        }
    }
    */
    input = TF_NewTensor(
        TF_FLOAT, dims,
        4, data, s*sizeof(float),
        &dealloc, NULL);


    if (!input)
    { 
        fprintf(stderr, "Failed to create input tensor\n");
        return std::vector<cv::Rect>();
    }
    clock_t t0 = clock();
    TF_SessionRun(sess,
                  NULL, // Run options.
                  &in_op, &input, 1, // Input tensors, input tensor values, number of inputs.
                  out_op, output, 2, // Output tensors, output tensor values, number of outputs.
                  NULL, 0, // Target operations, number of targets.
                  NULL, // Run metadata.
                  status // Output status.
    );
    printf("Inference alone took %.2f ms\n", 1000*((double)clock() - (double)t0)/CLOCKS_PER_SEC);
    int code = TF_GetCode(status);
    if (code)
    {
        const char* msg = TF_Message(status);
        fprintf(stderr, "\nSession run error! Status: %s\n\n", msg); 
        return std::vector<cv::Rect>();
    }
    
    if (!output[0])
    { 
        fprintf(stderr, "Failed to create output mask tensor!\n");
        return std::vector<cv::Rect>();
    }
    if (!output[1])
    { 
        fprintf(stderr, "Failed to create output bbox tensor!\n"); 
        return std::vector<cv::Rect>();
    }
    
    // ret_data is only n w*h binary masks. No channel dim
    int64_t* ret_data = (int64_t*)TF_TensorData(output[0]);
    
    // bounding box
    float* ret_data_bbox = (float*)TF_TensorData(output[1]);
   
    uint8_t* ret_data_uint8 = (uint8_t*)malloc(n*w*h*sizeof(uint8_t));
    std::copy(ret_data, ret_data + n*w*h, ret_data_uint8);

    /*
    for (int i = 0; i < h; i++)
    {
        for (int j = 0; j < w; j++)
            printf("%d ", ret_data_uint8[i*w + j]);
        printf("\n");
    }
    */

    
    //  save the generated mask into our return variable
    //  also copy the bounding box data to cv::Rect
    out.resize(n);
    std::vector<cv::Rect> bboxes(n);
    for (int i = 0; i < n; i++) 
    {   
        out[i] = cv::Mat(sz, CV_8UC1, ret_data_uint8 + i*h*w);
        bboxes[i] = cv::Rect(ret_data_bbox[i*4 + 0], ret_data_bbox[i*4 + 1], 
                        ret_data_bbox[i*4 + 2], ret_data_bbox[i*4 + 3]);
    }

    return bboxes;
}







