#pragma once

static const double LINE_DETECTOR_HOR_MSK[3][3] = { {-1.0,2.0,-1.0},
                                                {-1.0,2.0,-1.0},
                                                {-1.0,2.0,-1.0}
};

static const double LINE_DETECTOR_VER_MSK[3][3] = { {-1.0,-1.0,-1.0},
                                                   {2.0, 2.0, 2.0},
                                                   {-1.0,-1.0,-1.0}
};

static const double LINE_DETECTOR_LDIA_MSK[3][3] = { {2.0,-1.0,-1.0},
                                                   {-1.0,2.0,-1.0},
                                                   {-1.0,-1.0,2.0}
};
static const double LINE_DETECTOR_RDIA_MSK[3][3] = { {-1.0,-1.0,2.0},
                                                 {-1.0,2.0,-1.0},
                                                 {2.0,-1.0,-1.0}
};
static const double PREWITT_VERT[3][3] = {        {-1.0,0.0,1.0},
                                                 {-1.0,0.0,1.0},
                                                 {-1.0,0.0,1.0}
};
static const double PREWITT_HORIZ[3][3] = {       {-1.0,-1.0,-1.0},
                                                 {0.0,0.0,0.0},
                                                 {1.0,1.0,1.0}
};
static const double SOBEL_VERT[3][3] = {          {-1.0,0.0,1.0},
                                                 {-2.0,0.0,2.0},
                                                 {-1.0,0.0,1.0}
};
static const double SOBEL_HORIZ[3][3] = {         {-1.0,-2.0,-1.0},
                                                 {0.0,0.0,0.0},
                                                 {1.0,2.0,1.0}
};
static const double LAPLACIAN_NEG[3][3] = {      {0.0,-1.0,0.0},
                                                 {-1.0,4.0,-1.0},
                                                 {0.0,-1.0,0.0}
};
static const double lAPLACIAN_POS[3][3] = {      {0.0,1.0,0.0},
                                                 {1.0,-4.0,1.0},
                                                 {0.0,1.0,0.0}
};
static const double SHARPEN[3][3] = {            {-1.0,-1.0,-1.0},
                                                 {-1.0,9,-1.0},
                                                 {-1.0,-1.0,-1.0}
};
static const double TEST[3][3] = {               {-1.0,-1.0,-1.0},
                                                 {-1.0,8.0,-1.0},
                                                 {-1.0,-1.0,-1.0}
};
static const double BLUR[3][3] = {               {1/9.0,1/9.0,1/9.0},
                                                 {1/9.0,1/9.0,1/9.0},
                                                 {1/9.0,1/9.0,1/9.0}
};