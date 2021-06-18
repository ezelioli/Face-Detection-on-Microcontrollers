/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Main program body
  ******************************************************************************
  * @attention
  *
  * <h2><center>&copy; Copyright (c) 2021 STMicroelectronics.
  * All rights reserved.</center></h2>
  *
  * This software component is licensed by ST under BSD 3-Clause license,
  * the "License"; You may not use this file except in compliance with the
  * License. You may obtain a copy of the License at:
  *                        opensource.org/licenses/BSD-3-Clause
  *
  ******************************************************************************
  */
/* USER CODE END Header */
/* Includes ------------------------------------------------------------------*/
#include "main.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include <stdlib.h>
#include <stdio.h>
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"
#include "CycleCounter.h"
#include "onet.h"
//#include "rnet.h"
/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */
#define ARM_MATH_CM7
#define ARM_MATH_DSP
#define CMSIS_NN
#define TF_LITE_USE_GLOBAL_CMATH_FUNCTIONS
#define TF_LITE_USE_GLOBAL_MAX
#define TF_LITE_USE_GLOBAL_MIN
#define PUTCHAR_PROTOTYPE int __io_putchar(int ch)
/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/

UART_HandleTypeDef huart3;
GPIO_InitTypeDef GPIO_InitStruct;

/* USER CODE BEGIN PV */
namespace {
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* model_input = nullptr;
TfLiteTensor* model_output = nullptr;
TfLiteTensor* model_output_2 = nullptr;

  // Create an area of memory to use for input, output, and other TensorFlow
  // arrays. You'll need to adjust this by compiling, running, and looking
  // for errors.

} // namespace
int end;
int index_;
int frame_index;
uint8_t data_buffer[1];
uint8_t number[3];
//int8_t frame[24*24*3];
int8_t frame[48*48*3];
/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
static void MX_GPIO_Init(void);
static void MX_USART3_UART_Init(void);
/* USER CODE BEGIN PFP */
int8_t strtobyte(const char *str);
/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */
//int __io_putchar(int ch){
//	HAL_UART_Transmit(&huart3, (uint8_t *) &ch, 1, 0xFFFF);
//	return ch;
//}
PUTCHAR_PROTOTYPE
{
  HAL_UART_Transmit(&huart3, (uint8_t *) &ch, 1, 0xFFFF);
  return ch;
}
/* USER CODE END 0 */

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(void)
{
	/* USER CODE BEGIN 1 */
	char buf[8] = "ciao:)\n";
	char ch;
	char print_buf[50];
	int print_buf_len = 0;
	char err_buf[7] = "error\n";
	int err_buf_len = 6;
	TfLiteStatus tflite_status;
	const int kTensorArenaSize = 87 * 1024;
	//const int kTensorArenaSize = 20 * 1024;
	static uint8_t tensor_arena[kTensorArenaSize];
	uint32_t num_elements, num_output_elements;
	int8_t y_val, y_val_2;

	/* USER CODE END 1 */

	/* MCU Configuration--------------------------------------------------------*/

	/* Reset of all peripherals, Initializes the Flash interface and the Systick. */
	HAL_Init();

	/* USER CODE BEGIN Init */

	/* USER CODE END Init */

	/* Configure the system clock */
	SystemClock_Config();

	/* USER CODE BEGIN SysInit */

	/* USER CODE END SysInit */

	/* Initialize all configured peripherals */
	MX_GPIO_Init();
	MX_USART3_UART_Init();
	/* USER CODE BEGIN 2 */
	static tflite::MicroErrorReporter micro_error_reporter;
	error_reporter = &micro_error_reporter;
	print_buf_len = sprintf(print_buf, "START TEST\n");
	HAL_UART_Transmit(&huart3, (uint8_t *)print_buf, print_buf_len, 0xFFFF);
	// Map the model into a usable data structure
	model = tflite::GetModel(onet);
	if (model->version() != TFLITE_SCHEMA_VERSION)
	{
		HAL_UART_Transmit(&huart3, (uint8_t *)err_buf, err_buf_len, 0xFFFF);
		while(1);
	}
	//tflite::AllOpsResolver resolver;
	tflite::MicroMutableOpResolver<6> micro_op_resolver;
	//tflite::MicroMutableOpResolver<4>resolver;
	// Add dense neural network layer operation
	tflite_status = micro_op_resolver.AddFullyConnected();
	if (tflite_status != kTfLiteOk)
	{
		HAL_UART_Transmit(&huart3, (uint8_t *)err_buf, err_buf_len, 0xFFFF);
		while(1);
	}
	tflite_status = micro_op_resolver.AddConv2D();
	if (tflite_status != kTfLiteOk)
	{
		HAL_UART_Transmit(&huart3, (uint8_t *)err_buf, err_buf_len, 0xFFFF);
		while(1);
	}

	tflite_status =micro_op_resolver.AddReshape();
	if (tflite_status != kTfLiteOk)
	{
		HAL_UART_Transmit(&huart3, (uint8_t *)err_buf, err_buf_len, 0xFFFF);
		while(1);
	}
	tflite_status = micro_op_resolver.AddMaxPool2D();
	if (tflite_status != kTfLiteOk)
	{
		HAL_UART_Transmit(&huart3, (uint8_t *)err_buf, err_buf_len, 0xFFFF);
		while(1);
	}
	tflite_status = micro_op_resolver.AddRelu();
	if (tflite_status != kTfLiteOk)
	{
		HAL_UART_Transmit(&huart3, (uint8_t *)err_buf, err_buf_len, 0xFFFF);
		while(1);
	}
	tflite_status = micro_op_resolver.AddSoftmax();
	if (tflite_status != kTfLiteOk)
	{
		HAL_UART_Transmit(&huart3, (uint8_t *)err_buf, err_buf_len, 0xFFFF);
		while(1);
	}
	print_buf_len = sprintf(print_buf, "ALL ADDED RIGHT\r\n");
	HAL_UART_Transmit(&huart3, (uint8_t *)print_buf, print_buf_len, 100);
	//Build an interpreter to run the model with.
	print_buf_len = sprintf(print_buf, "Interpreter\r\n");
	HAL_UART_Transmit(&huart3, (uint8_t *)print_buf, print_buf_len, 100);
	static tflite::MicroInterpreter static_interpreter(
			model, micro_op_resolver, tensor_arena, kTensorArenaSize, error_reporter );
	interpreter = &static_interpreter;
	print_buf_len = sprintf(print_buf, "Successful\r\n");
	HAL_UART_Transmit(&huart3, (uint8_t *)print_buf, print_buf_len, 100);
	// Allocate memory from the tensor_arena for the model's tensors.
	print_buf_len = sprintf(print_buf, "Tensors\r\n");
	HAL_UART_Transmit(&huart3, (uint8_t *)print_buf, print_buf_len, 100);
	tflite_status = interpreter->AllocateTensors();
	if (tflite_status != kTfLiteOk)
	{
		print_buf_len = sprintf(print_buf, "Failed tensors\r\n");
		HAL_UART_Transmit(&huart3, (uint8_t *)print_buf, print_buf_len, 100);
		HAL_UART_Transmit(&huart3, (uint8_t *)err_buf, err_buf_len, 0xFFFF);
		while(1);
	}
	print_buf_len = sprintf(print_buf, "arena usage %d\n", interpreter->arena_used_bytes());
	HAL_UART_Transmit(&huart3, (uint8_t *)print_buf, print_buf_len, 100);
	print_buf_len = sprintf(print_buf, "Successful\r\n");
	HAL_UART_Transmit(&huart3, (uint8_t *)print_buf, print_buf_len, 100);
	// Assign model input and output buffers (tensors) to pointers
	model_input = interpreter->input(0);
	model_output = interpreter->output(0);
	model_output_2 = interpreter->output(1);
	float input_size = model_input->dims->size;
	print_buf_len = sprintf(print_buf, "Model input size: %d\r\n", (int)input_size);
	HAL_UART_Transmit(&huart3, (uint8_t *)print_buf, print_buf_len, 100);
	// Get number of elements in input tensor
	num_elements = model_input->bytes / sizeof(int) * 4;
	print_buf_len = sprintf(print_buf, "Number of input elements: %lu\r\n", num_elements);
	HAL_UART_Transmit(&huart3, (uint8_t *)print_buf, print_buf_len, 100);
//	float input_scale = model_input->params.scale;
//	int input_zero_point = model_input->params.zero_point;
	/* USER CODE END 2 */

	/* Infinite loop */
	/* USER CODE BEGIN WHILE */
	HAL_UART_Receive_IT (&huart3, data_buffer, 1);
	end = 0;
	index_ = 0;
	frame_index = 0;
	//HAL_GPIO_TogglePin(LED_BLUE_GPIO_Port, LED_BLUE_Pin);
//	for (uint32_t i = 0; i < num_elements; i++) {
//		model_input->data.int8[i] = frame[i];
//	}
	while (1)
	{
		if(end != 0){
			//HAL_GPIO_TogglePin(LED_BLUE_GPIO_Port, LED_BLUE_Pin);
			frame[frame_index] = strtobyte((char *)number);
			for(int i = 0; i < 3; ++i){
				number[i] = 0;
			}
			frame_index++;
			if(frame_index == 48 * 48 * 3){
				frame_index = 0;
				//HAL_GPIO_TogglePin(LED_BLUE_GPIO_Port, LED_BLUE_Pin);
				//HAL_GPIO_TogglePin(LED_RED_GPIO_Port, LED_RED_Pin);
				for (uint32_t i = 0; i < num_elements; i++) {
					model_input->data.int8[i] = frame[i];
				}
				tflite_status = interpreter->Invoke();
				y_val = model_output_2->data.int8[0];
				y_val_2 = model_output_2->data.int8[1];
				num_output_elements = model_output_2->bytes;
				if(y_val > y_val_2){
					HAL_GPIO_TogglePin(LED_RED_GPIO_Port, LED_RED_Pin);
				} else {
					HAL_GPIO_TogglePin(LED_BLUE_GPIO_Port, LED_BLUE_Pin);
				}
				HAL_Delay(1000);
				HAL_GPIO_WritePin(LED_RED_GPIO_Port, LED_RED_Pin, GPIO_PIN_RESET);
				HAL_GPIO_WritePin(LED_BLUE_GPIO_Port, LED_BLUE_Pin, GPIO_PIN_RESET);
			}
			end = 0;
		} else{
			__NOP();
		}
//		print_buf_len = sprintf(print_buf, "Trying to infer (NEW)\n");
//		HAL_UART_Transmit(&huart3, (uint8_t *)print_buf, print_buf_len, 100);
		// Fill input buffer (use test value)
//		for (uint32_t i = 0; i < num_elements; i++) {
//			model_input->data.int8[i] = image[i];
//		}
//		print_buf_len = sprintf(print_buf, "Timer reset\n");
//		HAL_UART_Transmit(&huart3, (uint8_t *)print_buf, print_buf_len, 100);
//		ResetTimer();
//		StartTimer();
//		tflite_status = interpreter->Invoke();
//		StopTimer();
//		print_buf_len = sprintf(print_buf, "Timer result: %u\n",getCycles());
//		HAL_UART_Transmit(&huart3, (uint8_t *)print_buf, print_buf_len, 100);
//		if(tflite_status != kTfLiteOk)
//		{
//			print_buf_len = sprintf(print_buf, "Invoke failed\r\n");
//			HAL_UART_Transmit(&huart3, (uint8_t *)print_buf, print_buf_len, 100);
//			error_reporter->Report("Invoke failed");
//		}
//		y_val = model_output_2->data.int8[0];
//		y_val_2 = model_output_2->data.int8[1];
//		num_output_elements = model_output_2->bytes;
//		print_buf_len = sprintf(print_buf, "Number of output elements: %lu\n", num_output_elements);
//		HAL_UART_Transmit(&huart3, (uint8_t *)print_buf, print_buf_len, 100);
//		print_buf_len = sprintf(print_buf, "Output: %d, %d\n", y_val, y_val_2);
//		HAL_UART_Transmit(&huart3, (uint8_t *)print_buf, print_buf_len, 100);
		/* USER CODE END WHILE */

		/* USER CODE BEGIN 3 */
//		HAL_GPIO_TogglePin(LED_BLUE_GPIO_Port, LED_BLUE_Pin);
		//HAL_Delay(2000);
	}
	/* USER CODE END 3 */
}

/**
  * @brief System Clock Configuration
  * @retval None
  */
void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};
  RCC_PeriphCLKInitTypeDef PeriphClkInitStruct = {0};

  /** Configure the main internal regulator output voltage
  */
  __HAL_RCC_PWR_CLK_ENABLE();
  __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE1);
  /** Initializes the RCC Oscillators according to the specified parameters
  * in the RCC_OscInitTypeDef structure.
  */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSI;
  RCC_OscInitStruct.HSIState = RCC_HSI_ON;
  RCC_OscInitStruct.HSICalibrationValue = RCC_HSICALIBRATION_DEFAULT;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSI;
  RCC_OscInitStruct.PLL.PLLM = 8;
  RCC_OscInitStruct.PLL.PLLN = 216;
  RCC_OscInitStruct.PLL.PLLP = RCC_PLLP_DIV2;
  RCC_OscInitStruct.PLL.PLLQ = 4;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }
  /** Activate the Over-Drive mode
  */
  if (HAL_PWREx_EnableOverDrive() != HAL_OK)
  {
    Error_Handler();
  }
  /** Initializes the CPU, AHB and APB buses clocks
  */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV4;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV2;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_7) != HAL_OK)
  {
    Error_Handler();
  }
  PeriphClkInitStruct.PeriphClockSelection = RCC_PERIPHCLK_USART3;
  PeriphClkInitStruct.Usart3ClockSelection = RCC_USART3CLKSOURCE_PCLK1;
  if (HAL_RCCEx_PeriphCLKConfig(&PeriphClkInitStruct) != HAL_OK)
  {
    Error_Handler();
  }
}

/**
  * @brief USART3 Initialization Function
  * @param None
  * @retval None
  */
static void MX_USART3_UART_Init(void)
{

  /* USER CODE BEGIN USART3_Init 0 */

  /* USER CODE END USART3_Init 0 */

  /* USER CODE BEGIN USART3_Init 1 */

  /* USER CODE END USART3_Init 1 */
  huart3.Instance = USART3;
  huart3.Init.BaudRate = 115200;
  huart3.Init.WordLength = UART_WORDLENGTH_8B;
  huart3.Init.StopBits = UART_STOPBITS_1;
  huart3.Init.Parity = UART_PARITY_NONE;
  huart3.Init.Mode = UART_MODE_TX_RX;
  huart3.Init.HwFlowCtl = UART_HWCONTROL_NONE;
  huart3.Init.OverSampling = UART_OVERSAMPLING_16;
  huart3.Init.OneBitSampling = UART_ONE_BIT_SAMPLE_DISABLE;
  huart3.AdvancedInit.AdvFeatureInit = UART_ADVFEATURE_NO_INIT;
  if (HAL_UART_Init(&huart3) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN USART3_Init 2 */

  /* USER CODE END USART3_Init 2 */

}

/**
  * @brief GPIO Initialization Function
  * @param None
  * @retval None
  */
static void MX_GPIO_Init(void)
{
  GPIO_InitTypeDef GPIO_InitStruct = {0};

  /* GPIO Ports Clock Enable */
  __HAL_RCC_GPIOB_CLK_ENABLE();
  __HAL_RCC_GPIOD_CLK_ENABLE();

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(GPIOB, LED_RED_Pin|LED_BLUE_Pin, GPIO_PIN_RESET);

  /*Configure GPIO pins : LED_RED_Pin LED_BLUE_Pin */
  GPIO_InitStruct.Pin = LED_RED_Pin|LED_BLUE_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(GPIOB, &GPIO_InitStruct);

}

/* USER CODE BEGIN 4 */
void HAL_UART_RxCpltCallback(UART_HandleTypeDef *huart)
{
	UNUSED(huart);
	if(data_buffer[0] != '\n'){
		number[index_] = data_buffer[0];
		index_++;
	}else{
		index_ = 0;
		end = 1;
	}
	HAL_UART_Receive_IT (&huart3, data_buffer, 1);
}

int8_t strtobyte(const char *str){
	int8_t b = 0;

	b = (int8_t) (atoi(str) - 128);

	return b;
}
/* USER CODE END 4 */

/**
  * @brief  This function is executed in case of error occurrence.
  * @retval None
  */
void Error_Handler(void)
{
  /* USER CODE BEGIN Error_Handler_Debug */
  /* User can add his own implementation to report the HAL error return state */
  __disable_irq();
  while (1)
  {
  }
  /* USER CODE END Error_Handler_Debug */
}

#ifdef  USE_FULL_ASSERT
/**
  * @brief  Reports the name of the source file and the source line number
  *         where the assert_param error has occurred.
  * @param  file: pointer to the source file name
  * @param  line: assert_param error line source number
  * @retval None
  */
void assert_failed(uint8_t *file, uint32_t line)
{
  /* USER CODE BEGIN 6 */
  /* User can add his own implementation to report the file name and line number,
     ex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */
  /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */

/************************ (C) COPYRIGHT STMicroelectronics *****END OF FILE****/
