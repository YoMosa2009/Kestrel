using System.Windows;

namespace Kestrel.Studio;

public partial class App : Application
{
    protected override void OnStartup(StartupEventArgs e)
    {
        DispatcherUnhandledException += (_, ex) =>
        {
            MessageBox.Show(ex.Exception.Message, "Kestrel Studio", MessageBoxButton.OK, MessageBoxImage.Error);
            ex.Handled = true;
        };
        base.OnStartup(e);
    }
}
