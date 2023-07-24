package ai.mlc.mlcchat

import ai.mlc.mlcchat.ui.theme.MLCChatTheme
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Surface
import androidx.compose.ui.Modifier
import android.app.ActivityManager
import android.content.Context
import android.os.Debug
import kotlinx.coroutines.GlobalScope
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import android.util.Log

var TAG="MONITOR"
fun startMemoryMonitoring(context: Context) {
    GlobalScope.launch {
        while (true) {
            // 获取 ActivityManager 和 Debug.MemoryInfo 对象
            val activityManager = context.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
            var memory = activityManager.memoryClass
            var maxMemory = Runtime.getRuntime().maxMemory() * 1.0 / (1024*1024)
            var totalMemory = Runtime.getRuntime().totalMemory() * 1.0 / (1024*1024)
            var freeMemory = Runtime.getRuntime().freeMemory() * 1.0 / (1024*1024)
            val memoryInfo = ActivityManager.MemoryInfo()
            activityManager.getMemoryInfo(memoryInfo)

            // 打印可用内存和应用程序内存占用情况
            Log.i(TAG,"======================================================")
            Log.i(TAG,"总内存：${memoryInfo.totalMem / (1024 * 1024)} MB")
            Log.i(TAG,"可用内存：${memoryInfo.availMem / (1024 * 1024)} MB")
            Log.i(TAG, "应用程序内存占用：${getAppMemoryUsage(context) / (1024 * 1024)} MB")
            Log.i(TAG, "最大内存：${memory} MB")
            Log.i(TAG, "最大内存：${maxMemory} MB")
            Log.i(TAG, "app内存总占用：${getAppMemoryUsage(context) / (1024 * 1024)} MB")
            Log.i(TAG, "app内存空闲：${getAppMemoryUsage(context) / (1024 * 1024)} MB")

            // 延迟 1 秒后继续监测
            delay(500)
        }
    }
}

fun getAppMemoryUsage(context: Context): Long {
    val memoryInfo = Debug.MemoryInfo()
    Debug.getMemoryInfo(memoryInfo)
    return memoryInfo.totalPrivateDirty.toLong()
}

class MainActivity : ComponentActivity() {

    @ExperimentalMaterial3Api
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            Surface(
                modifier = Modifier
                    .fillMaxSize()
            ) {
                MLCChatTheme {
                    NavView()
                }
            }
        }
//        var context: Context = applicationContext
//        startMemoryMonitoring(context)
    }
}